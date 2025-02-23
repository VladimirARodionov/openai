#!/bin/sh

# Проверяем, передан ли путь к бэкапу
if [ -z "$1" ]; then
    echo "Usage: $0 <backup_archive>"
    echo "Available backups:"
    ls -1 /opt/couchbase/backup/*.tar.gz
    exit 1
fi

ARCHIVE_PATH="/opt/couchbase/backup/$1"
TEMP_RESTORE_DIR="/opt/couchbase/backup/temp_restore"

# Проверяем существование архива
if [ ! -f "${ARCHIVE_PATH}" ]; then
    echo "Backup archive ${ARCHIVE_PATH} not found"
    exit 1
fi

echo "Starting restore from ${ARCHIVE_PATH}..."

# Создаем временную директорию и распаковываем архив
mkdir -p "${TEMP_RESTORE_DIR}"
tar -xzf "${ARCHIVE_PATH}" -C "${TEMP_RESTORE_DIR}"

if [ $? -ne 0 ]; then
    echo "Failed to extract backup archive"
    rm -rf "${TEMP_RESTORE_DIR}"
    exit 1
fi

# Очищаем существующий bucket через N1QL
echo "Clearing existing bucket..."
/opt/couchbase/bin/cbq -e localhost:8093 \
    -u ${COUCHBASE_ADMINISTRATOR_USERNAME} \
    -p ${COUCHBASE_ADMINISTRATOR_PASSWORD} \
    --script="DELETE FROM vector_store._default._default"

# Выполняем восстановление
/opt/couchbase/bin/cbbackupmgr restore \
    --archive ${TEMP_RESTORE_DIR} \
    --repo backup_repository \
    --cluster localhost:8091 \
    --username ${COUCHBASE_ADMINISTRATOR_USERNAME} \
    --password ${COUCHBASE_ADMINISTRATOR_PASSWORD} \
    --no-ssl-verify

RESTORE_STATUS=$?

# Очищаем временную директорию
rm -rf "${TEMP_RESTORE_DIR}"

if [ ${RESTORE_STATUS} -eq 0 ]; then
    echo "Restore completed successfully"
else
    echo "Restore failed"
    exit 1
fi 