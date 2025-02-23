#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

BACKUP_FILE="/backups/$1"

if [ ! -f "${BACKUP_FILE}" ]; then
    echo "Backup file not found: ${BACKUP_FILE}"
    exit 1
fi

# Восстанавливаем из бэкапа
echo "Dropping existing database..."
dropdb -U ${POSTGRES_USER} ${POSTGRES_DB} --if-exists
createdb -U ${POSTGRES_USER} ${POSTGRES_DB}

echo "Restoring from backup..."
gunzip -c ${BACKUP_FILE} | psql -U ${POSTGRES_USER} ${POSTGRES_DB}

if [ $? -eq 0 ]; then
    echo "Restore completed successfully"
else
    echo "Error during restore"
    exit 1
fi 