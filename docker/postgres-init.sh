#!/bin/sh
# Create the Airflow metadata database.
#
# The compose file points Airflow at postgres/airflow, but the postgres image
# only creates the one database named in POSTGRES_DB (platform). Nothing created
# the second one, so `airflow standalone` failed at startup against a database
# that did not exist. This runs from /docker-entrypoint-initdb.d on first boot.
#
# Only on FIRST boot, when the data directory is empty. On an existing volume
# this script is not run, so a cluster created before it existed needs
# `docker compose down -v` or a manual CREATE DATABASE.
set -eu

psql --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<SQL
    SELECT 'CREATE DATABASE airflow'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'airflow')\gexec
SQL
