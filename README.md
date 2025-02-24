# openai

## config
add .env file as copy of .env.example file

chmod -R a+rw logs

sudo docker compose build --no-cache

sudo docker compose up --build -d

## Создание бэкапа
docker compose exec couchbase manage_backups backup

docker compose exec postgres manage_backups backup

## Восстановление
docker compose exec couchbase manage_backups restore backup_20240101_120000.tar.gz

docker compose exec postgres manage_backups restore pg_backup_20240315_120000.sql.gz

## Список бэкапов
docker compose exec couchbase manage_backups list

docker compose exec postgres manage_backups list