#!/bin/bash

docker run -d -p 5000:5000 --env-file .env --name powerstocksearch-app powerstocksearch
