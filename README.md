# Set OpenSearch Docker
```sh
docker pull opensearchproject/opensearch:latest
docker run -d --name opensearch -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" -e "DISABLE_SECURITY_PLUGIN=true" opensearchproject/opensearch:latest
```

# Run OpenSearch Docker
```sh
docker start opensearch
```

# Run UI/UX
```sh
cd frontend
npm run dev
```