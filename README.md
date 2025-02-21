# openai

## config
add .env file as copy of .env.example file

source venv/bin/activate

pip install -r requirements.txt

alembic upgrade head

chmod -R +x init-scripts

chmod -R a+rw logs

sudo docker compose -p ay build --no-cache

sudo docker compose -p ay up --build -d

python bot.py

## Создание бэкапа
docker exec couchbase /docker-entrypoint-initdb.d/manage_backups.sh backup

## Восстановление
docker exec couchbase /docker-entrypoint-initdb.d/manage_backups.sh restore backup_20240101_120000.tar.gz

## Список бэкапов
docker exec couchbase /docker-entrypoint-initdb.d/manage_backups.sh list