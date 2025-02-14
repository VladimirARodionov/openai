#!/bin/sh

# Проверяем, был ли уже инициализирован Couchbase
INIT_FLAG="/opt/couchbase/init_done/.initialized"

if [ -f "$INIT_FLAG" ]; then
    echo "Couchbase already initialized, skipping initialization"
    exit 0
fi

# Функция проверки готовности сервиса
wait_for_service() {
    local service=$1
    local max_attempts=30
    local attempt=1
    
    echo "Waiting for $service service..."
    while [ $attempt -le $max_attempts ]; do
        # Проверяем наличие сервиса в массиве services
        if curl -s http://localhost:8091/pools/default | grep -q "\"services\".*\"$service\""; then
            echo "$service service is ready"
            return 0
        fi
        echo "Waiting for $service service... ($attempt/$max_attempts)"
        sleep 5
        attempt=$((attempt + 1))
    done
    echo "$service service failed to start"
    return 1
}

# Функция проверки существования bucket
wait_for_bucket() {
    local bucket=$1
    local max_attempts=30
    local attempt=1
    
    echo "Waiting for bucket $bucket..."
    while [ $attempt -le $max_attempts ]; do
        if curl -s -u ${COUCHBASE_ADMINISTRATOR_USERNAME}:${COUCHBASE_ADMINISTRATOR_PASSWORD} \
            http://localhost:8091/pools/default/buckets/$bucket > /dev/null; then
            echo "Bucket $bucket is ready"
            return 0
        fi
        echo "Waiting for bucket $bucket... ($attempt/$max_attempts)"
        sleep 5
        attempt=$((attempt + 1))
    done
    echo "Bucket $bucket failed to create"
    return 1
}

# Функция проверки статуса ребалансировки
wait_for_rebalance() {
    local max_attempts=60  # Увеличиваем максимальное количество попыток
    local attempt=1
    
    echo "Waiting for rebalance to complete..."
    while [ $attempt -le $max_attempts ]; do
        # Проверяем статус ребалансировки
        local status=$(curl -s -u ${COUCHBASE_ADMINISTRATOR_USERNAME}:${COUCHBASE_ADMINISTRATOR_PASSWORD} \
            http://localhost:8091/pools/default/rebalanceProgress)
        
        # Проверяем статус индексов
        local index_status=$(curl -s -u ${COUCHBASE_ADMINISTRATOR_USERNAME}:${COUCHBASE_ADMINISTRATOR_PASSWORD} \
            http://localhost:8091/indexStatus)
        
        if echo "$status" | grep -q '"status":"none"' && \
           ! echo "$index_status" | grep -q '"status":"rebalancing"'; then
            echo "Rebalance and index rebalancing completed"
            # Дополнительная пауза для стабилизации
            sleep 10
            return 0
        fi
        echo "Waiting for rebalance... ($attempt/$max_attempts)"
        sleep 5
        attempt=$((attempt + 1))
    done
    echo "Rebalance failed to complete"
    return 1
}

# Функция проверки статуса индекса
wait_for_index() {
    local index_name=$1
    local max_attempts=30
    local attempt=1
    
    echo "Waiting for index $index_name to become online..."
    while [ $attempt -le $max_attempts ]; do
        # Проверяем статус индекса через N1QL с явным указанием scope
        local status=$(/opt/couchbase/bin/cbq -e localhost:8093 \
            -u ${COUCHBASE_ADMINISTRATOR_USERNAME} \
            -p ${COUCHBASE_ADMINISTRATOR_PASSWORD} \
            --quiet \
            --script="SELECT state FROM system:indexes WHERE name = '$index_name' AND keyspace_id = 'vector_store' AND scope_id = '_default';")
        
        if echo "$status" | grep -q '"state": "online"'; then
            echo "Index $index_name is online"
            return 0
        fi
        echo "Waiting for index $index_name... ($attempt/$max_attempts)"
        sleep 5
        attempt=$((attempt + 1))
    done
    echo "Index $index_name failed to become online"
    return 1
}

# Ждем запуска Couchbase
while ! curl -s http://localhost:8091 > /dev/null; do
    echo "Waiting for Couchbase to start..."
    sleep 3
done

echo "Initializing Couchbase..."

# Инициализация кластера с включенным Search сервисом
/opt/couchbase/bin/couchbase-cli cluster-init \
  -c localhost \
  --cluster-username ${COUCHBASE_ADMINISTRATOR_USERNAME} \
  --cluster-password ${COUCHBASE_ADMINISTRATOR_PASSWORD} \
  --services data,index,query,fts \
  --cluster-ramsize 1024 \
  --cluster-index-ramsize 512 \
  --cluster-fts-ramsize 512

# Ждем готовности всех сервисов
#wait_for_service kv || exit 1    # data service называется kv в API
#wait_for_service n1ql || exit 1  # query service называется n1ql в API
#wait_for_service index || exit 1
#wait_for_service fts || exit 1


sleep 10

# Создание bucket
/opt/couchbase/bin/couchbase-cli bucket-create \
  -c localhost \
  --username ${COUCHBASE_ADMINISTRATOR_USERNAME} \
  --password ${COUCHBASE_ADMINISTRATOR_PASSWORD} \
  --bucket vector_store \
  --bucket-type couchbase \
  --bucket-ramsize 512

# Ждем создания bucket
wait_for_bucket vector_store || exit 1

# Увеличиваем время ожидания после создания бакета
sleep 10

# Ждем завершения ребалансировки
wait_for_rebalance || exit 1

# Дополнительная пауза перед созданием индексов
sleep 10

# Создание первичного индекса
/opt/couchbase/bin/cbq -e localhost:8093 -u ${COUCHBASE_ADMINISTRATOR_USERNAME} -p ${COUCHBASE_ADMINISTRATOR_PASSWORD} \
  --script="CREATE PRIMARY INDEX ON vector_store._default._default USING GSI;"

# Ждем, пока первичный индекс станет онлайн
#wait_for_index "#primary" || exit 1

# Создание FTS индекса для векторного поиска
curl -X PUT -u ${COUCHBASE_ADMINISTRATOR_USERNAME}:${COUCHBASE_ADMINISTRATOR_PASSWORD} \
  http://localhost:8094/api/bucket/vector_store/scope/_default/index/vector-index \
  -H "Content-Type: application/json" \
  -d '{
       "name": "vector-index",
       "sourceName": "vector_store",
       "type": "fulltext-index",
       "sourceParams": {
         "scope": "_default",
         "collection": "_default"
       },
       "params": {
        "doc_config": {
         "docid_prefix_delim": "",
         "docid_regexp": "",
         "mode": "type_field",
         "type_field": "type"
        },
        "mapping": {
         "default_analyzer": "standard",
         "default_datetime_parser": "dateTimeOptional",
         "default_field": "_all",
         "default_mapping": {
          "dynamic": true,
          "enabled": true,
          "properties": {
           "metadata": {
            "dynamic": true,
            "enabled": true
           },
           "embedding": {
            "enabled": true,
            "dynamic": false,
            "fields": [
             {
              "dims": 1536,
              "index": true,
              "name": "embedding",
              "similarity": "dot_product",
              "type": "vector",
              "vector_index_optimized_for": "recall"
             }
            ]
           },
           "text": {
            "enabled": true,
            "dynamic": false,
            "fields": [
             {
              "index": true,
              "name": "text",
              "store": true,
              "type": "text"
             }
            ]
           }
          }
         },
         "default_type": "_default",
         "docvalues_dynamic": false,
         "index_dynamic": true,
         "store_dynamic": true,
         "type_field": "_type"
        },
        "store": {
         "indexType": "scorch",
         "segmentVersion": 16
        }
       },
       "sourceType": "couchbase",
       "planParams": {
        "maxPartitionsPerPIndex": 103,
        "indexPartitions": 10,
        "numReplicas": 0
       }
      }'


# Проверяем создание индексов
echo "Checking GSI indexes..."
/opt/couchbase/bin/cbq -e localhost:8093 -u ${COUCHBASE_ADMINISTRATOR_USERNAME} -p ${COUCHBASE_ADMINISTRATOR_PASSWORD} \
  --script="SELECT * FROM system:indexes WHERE keyspace_id = 'vector_store';"

# Создаем флаг инициализации
mkdir -p "$(dirname "$INIT_FLAG")"
touch "$INIT_FLAG"

echo "Couchbase initialization completed" 