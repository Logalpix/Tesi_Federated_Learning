# Avvia il nodo master
echo "Avvio nodo master..."
docker-compose up -d master

# Recupera l'indirizzo IP del nodo master
MASTER_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' master_node)

# Attendi che il master generi il Peer ID
echo "Ottenimento peer ID master..."
MASTER_PEER_ID=""
while [ -z "$MASTER_PEER_ID" ]; do
  MASTER_PEER_ID=$(docker logs master_node 2>&1 | grep -oE '/p2p/([A-Za-z0-9]+)')
  sleep 1
done

echo "Peer ID Master ottenuto: $MASTER_PEER_ID"

# Aggiorna l'indirizzo del master nei comandi dei worker
MASTER_ADDR='//ip4/'${MASTER_IP}'/tcp/4000'${MASTER_PEER_ID}
export MASTER_ADDR

echo "MASTER_ADDR impostata a $MASTER_ADDR"

# Avvia i nodi worker con il Peer ID corretto
echo "Avvio nodi worker con MASTER_ADDR=${MASTER_ADDR}..."
docker-compose up -d worker

echo "Avvio dei nodi effettuato!"