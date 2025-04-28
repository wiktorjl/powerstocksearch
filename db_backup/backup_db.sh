#!/bin/sh

echo "Grabbing the schema."
docker exec d5 pg_dump -U pricepirate -d stonks --schema-only > schema.sql
echo "Grabbing the data."
docker exec d5 pg_dump -U pricepirate -d stonks --data-only   > data.sql

echo "Compressing SQL scripts."
tar czfv "backup_$(date +%Y%m%d_%H%M%S).tar.gz" schema.sql data.sql

echo "Removing sql files."
rm -i schema.sql
rm -i data.sql

echo 
echo "DONE!"
echo
