# Изменение размера бакета Couchbase

## Способ 1: Через docker-compose.yml

```yaml
environment:
  - COUCHBASE_BUCKET_RAMSIZE=1024  # Увеличить до 1GB
```

## Способ 2: Через CLI (после запуска)

```bash
# Подключиться к контейнеру
docker-compose exec couchbase bash

# Изменить размер бакета
/opt/couchbase/bin/couchbase-cli bucket-edit \
  -c localhost \
  --username admin \
  --password password \
  --bucket vector_store \
  --bucket-ramsize 1024
```

## Способ 3: Через веб-интерфейс
1. Откройте http://localhost:8091
2. Войдите с admin/password
3. Buckets → vector_store → Edit
4. Измените RAM Quota

## Рекомендации по размеру:

- **Минимум**: 512 МБ (текущий)
- **Рекомендуется**: 1-2 ГБ для среднего объема данных
- **Большие данные**: 4+ ГБ для больших коллекций документов

⚠️ **Внимание**: После изменения размера нужно перезапустить контейнер
