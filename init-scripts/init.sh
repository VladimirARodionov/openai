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
sleep 10

# Создание первичного индекса
/opt/couchbase/bin/cbq -e localhost:8093 -u ${COUCHBASE_ADMINISTRATOR_USERNAME} -p ${COUCHBASE_ADMINISTRATOR_PASSWORD} \
  --script="CREATE PRIMARY INDEX ON \`vector_store\`;"

# Создание индекса для embedding
/opt/couchbase/bin/cbq -e localhost:8093 -u ${COUCHBASE_ADMINISTRATOR_USERNAME} -p ${COUCHBASE_ADMINISTRATOR_PASSWORD} \
  --script="CREATE INDEX embedding_index ON \`vector_store\`(embedding) USING GSI;"

# Создаем флаг инициализации
mkdir -p "$(dirname "$INIT_FLAG")"
touch "$INIT_FLAG"

echo "Couchbase initialization completed" 