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
docker exec ay-couchbase-1 manage_backups.sh backup

docker exec ay-postgres-1 manage_backups.sh backup

## Восстановление
docker exec ay-couchbase-1 manage_backups.sh restore backup_20240101_120000.tar.gz

docker exec ay-postgres-1 manage_backups.sh restore pg_backup_20240315_120000.sql.gz

## Список бэкапов
docker exec ay-couchbase-1 manage_backups.sh list

docker exec ay-postgres-1 manage_backups.sh list