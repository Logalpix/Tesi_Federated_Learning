Il progetto consiste in un'implementazione di federated learning (come alternativa centralizzata al gossip learning) utilizzando Node.js, PyTorch e Docker.

Per avviare il progetto è necessaria l'installazione sul sistema di Docker, Docker Compose, Node.js (versione 20 o superiore), Python, PyTorch e, opzionalmente, CUDA 11.8.

La rete neurale è costituita da 8 nodi worker e 1 nodo master, istanziati come container Docker e definiti nel file docker-compose.yml.
Attualmente, il dataset su cui viene effettuato l'apprendimento è GTSRB, definito all'interno del codice python nel file federated_learning.js.
Per eseguire codice python su Node.js è stata usata la libreria python-bridge.
Per la comunicazione tra i nodi è stata usata la libreria libp2p.

Per avviare l'apprendimento occorre lanciare da bash il comando docker-compose build e in seguito lo script ./start.sh, che si occuperà di avviare prima il nodo master per ottenerne l'indirizzo e in seguito i nodi worker, passando l'indirizzo del master come parametro. Per fermare e rimuovere i container occorre lanciare il comando docker-compose down.

Durante la build verranno scaricati sia i moduli di Node.js necessari per il progetto, sia il dataset su cui viene effettuato il training, salvati rispettivamente nelle cartelle node_modules/ e data/ interne ai container. Nella stessa locazione verrà inoltre creata una cartella models/, dove vengono salvati i modelli su cui i nodi effettuano il training.