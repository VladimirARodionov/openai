#!/bin/sh

# Функция проверки готовности Couchbase
wait_for_couchbase() {
    echo "Waiting for Couchbase to be ready..."
    for i in $(seq 1 30); do
        if curl -s http://localhost:8091/pools/default > /dev/null; then
            echo "Couchbase is ready"
            return 0
        fi
        echo "Waiting... ($i/30)"
        sleep 5
    done
    echo "Couchbase failed to start within timeout"
    return 1
}

# Ждем готовности Couchbase
wait_for_couchbase || exit 1

# Получаем текущую дату для имени бэкапа
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/opt/couchbase/backup/temp_backup_${BACKUP_DATE}"
ARCHIVE_NAME="/opt/couchbase/backup/backup_${BACKUP_DATE}.tar.gz"

echo "Starting backup to ${BACKUP_DIR}..."

# Создаем временную директорию для бэкапа и репозиторий
mkdir -p "${BACKUP_DIR}"

# Инициализируем репозиторий бэкапа
/opt/couchbase/bin/cbbackupmgr config \
    --archive ${BACKUP_DIR} \
    --repo backup_repository

# Выполняем бэкап
/opt/couchbase/bin/cbbackupmgr backup \
    --archive ${BACKUP_DIR} \
    --repo backup_repository \
    --cluster localhost:8091 \
    --username ${COUCHBASE_ADMINISTRATOR_USERNAME} \
    --password ${COUCHBASE_ADMINISTRATOR_PASSWORD} \
    --no-ssl-verify

if [ $? -eq 0 ]; then
    # Создаем файл с метаданными бэкапа
    echo "Backup created at: $(date)" > "${BACKUP_DIR}/backup_info.txt"
    echo "Bucket: vector_store" >> "${BACKUP_DIR}/backup_info.txt"
    
    # Архивируем бэкап
    tar -czf "${ARCHIVE_NAME}" -C "${BACKUP_DIR}" .
    
    if [ $? -eq 0 ]; then
        echo "Backup completed and archived successfully to ${ARCHIVE_NAME}"
        # Удаляем временную директорию
        rm -rf "${BACKUP_DIR}"
    else
        echo "Backup completed but archiving failed"
        exit 1
    fi
else
    echo "Backup failed"
    rm -rf "${BACKUP_DIR}"
    exit 1
fi 