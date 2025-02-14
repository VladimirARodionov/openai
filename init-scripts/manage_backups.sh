#!/bin/sh

# Функция для вывода помощи
show_help() {
    echo "Usage:"
    echo "  $0 backup                    - Create new backup"
    echo "  $0 restore <backup_name>     - Restore from backup"
    echo "  $0 list                      - List available backups"
    echo "  $0 delete <backup_name>      - Delete specific backup"
    echo "  $0 delete-old <days>         - Delete backups older than X days"
}

BACKUP_BASE_DIR="/opt/couchbase/backup"
SCRIPTS_DIR="/docker-entrypoint-initdb.d"

case "$1" in
    backup)
        ${SCRIPTS_DIR}/backup.sh
        ;;
    restore)
        if [ -z "$2" ]; then
            echo "Error: Backup name required"
            show_help
            exit 1
        fi
        ${SCRIPTS_DIR}/restore.sh "$2"
        ;;
    list)
        echo "Available backups:"
        ls -lh "${BACKUP_BASE_DIR}"/*.tar.gz 2>/dev/null || echo "No backups found"
        ;;
    delete)
        if [ -z "$2" ]; then
            echo "Error: Backup name required"
            show_help
            exit 1
        fi
        rm -f "${BACKUP_BASE_DIR}/$2"
        echo "Backup $2 deleted"
        ;;
    delete-old)
        if [ -z "$2" ]; then
            echo "Error: Number of days required"
            show_help
            exit 1
        fi
        find "${BACKUP_BASE_DIR}" -name "*.tar.gz" -type f -mtime +$2 -exec rm -f {} \;
        echo "Old backups deleted"
        ;;
    *)
        show_help
        exit 1
        ;;
esac 