#!/bin/sh

# Проверяем, был ли уже инициализирован Couchbase
INIT_FLAG="/opt/couchbase/init_done/.initialized"

if [ -f "$INIT_FLAG" ]; then
    echo "Couchbase already initialized, skipping initialization"
    exit 0
fi

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

# Ждем инициализации кластера
sleep 5

# Создание bucket
/opt/couchbase/bin/couchbase-cli bucket-create \
  -c localhost \
  --username ${COUCHBASE_ADMINISTRATOR_USERNAME} \
  --password ${COUCHBASE_ADMINISTRATOR_PASSWORD} \
  --bucket vector_store \
  --bucket-type couchbase \
  --bucket-ramsize 512

# Ждем создания bucket
sleep 5

# Создание первичного индекса
/opt/couchbase/bin/cbq -e localhost:8093 -u ${COUCHBASE_ADMINISTRATOR_USERNAME} -p ${COUCHBASE_ADMINISTRATOR_PASSWORD} \
  --script="CREATE PRIMARY INDEX ON \`vector_store\`;"

# Создание векторного индекса
/opt/couchbase/bin/cbq -e localhost:8093 -u ${COUCHBASE_ADMINISTRATOR_USERNAME} -p ${COUCHBASE_ADMINISTRATOR_PASSWORD} \
  --script="
  CREATE SEARCH INDEX vector_index ON \`vector_store\`(embedding) 
  USING FTS 
  WITH {\"type\": \"vector\", \"dims\": 1536};"

# Создаем флаг инициализации
mkdir -p "$(dirname "$INIT_FLAG")"
touch "$INIT_FLAG"

echo "Couchbase initialization completed" 