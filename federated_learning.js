import process from 'node:process'
import { createLibp2p } from 'libp2p'
import { tcp } from '@libp2p/tcp'
import { mplex } from '@libp2p/mplex'
import { noise } from '@chainsafe/libp2p-noise'
import { peerIdFromString } from '@libp2p/peer-id'
import { multiaddr } from 'multiaddr'
import { mdns } from '@libp2p/mdns'
import { pipe } from 'it-pipe'
import toBuffer from 'it-to-buffer'
import pythonBridge from 'python-bridge'

import * as os from "os"
import fs from 'fs'
import { createHash } from 'node:crypto'

import AsyncLock from 'async-lock'

const lock = new AsyncLock()
var num_of_models_received = 0
const path_dir_models = 'models/'
var master_id

var peer_id_known_peers = []

function get_position_str(string, subString, index) {
	return string.split(subString, index).join(subString).length
}

function get_ip_addr(){
	const interfaces = os.networkInterfaces();
	for (const name of Object.keys(interfaces)) {
			for (const net of interfaces[name]) {
					// Skip over non-IPv4 and internal (i.e., 127.0.0.1) addresses
					if (net.family === 'IPv4' && !net.internal) {
							return net.address;
					}
			}
	}
	throw new Error('Nessun indirizzo IP valido trovato');
}

function get_peerid_from_multiadd(multiadd){
	//return multiadd.substring(get_position_str(multiadd, '/', 6)+1, get_position_str(multiadd, '/', 7))
	return multiadd.substring(multiadd.indexOf('/p2p/') + 5);
}

function create_random_str(length) {
		let result = ''
		const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
		const charactersLength = characters.length
		let counter = 0
		while (counter < length) {
			result += characters.charAt(Math.floor(Math.random() * charactersLength))
			counter += 1
		}
		return result
}

function sha256(content) {  
	return createHash('sha256').update(content).digest('hex')
}

function delay(time) {
	return new Promise(resolve => setTimeout(resolve, time))
} 

async function on_model_received_master ({ stream }) {
	console.log("on_model_received_master invocato.")
	const communication_content = await pipe(
		stream,
		async function * (source) {
			for await (const list of source) {
				yield list.subarray()
			}
		},
		toBuffer
		).finally(() => {
		stream.close()
		})
	console.log('sha256 del file modello ricevuto: ', sha256(communication_content))

	var random_name_file = create_random_str(10) + '.pt'
	fs.writeFileSync(path_dir_models + random_name_file, communication_content)
	
	lock.acquire('num_of_models_received', function(){
		num_of_models_received = num_of_models_received + 1
	})
	
}

async function on_model_received_worker ({ stream }) {
	console.log("on_model_received_worker invocato.")
	const communication_content = await pipe(
		stream,
		async function * (source) {
			for await (const list of source) {
				yield list.subarray()
			}
		},
		toBuffer
		).finally(() => {
		stream.close()
		})
	
	var random_name_file = create_random_str(10) + '.pt'
	console.log('sha256 del file modello ricevuto: ', sha256(communication_content))
	fs.writeFileSync(path_dir_models + random_name_file, communication_content)
	
	await python.ex`
	print("loading model received")
	local_model.load_state_dict(torch.load(${path_dir_models} + ${random_name_file}))
	print("training local model")
		
	client_update(local_model, opt, train_loader, epoch=epochs)
	torch.save(local_model.state_dict(), ${path_dir_models} + ${random_name_file})
	`
	
	const content_model_file = await fs.readFileSync(path_dir_models + random_name_file)
	const stream_to_master = await node.dialProtocol(peerIdFromString(master_id), '/on_model_received_master')
	await pipe([content_model_file], stream_to_master)
	
	fs.unlinkSync(path_dir_models + random_name_file)
	
	console.log('sha256 del file modello inviato: ', sha256(content_model_file))
}

var python = pythonBridge({
		python: 'python'
})

python.ex`
import math
import random
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset   
torch.backends.cudnn.benchmark=True

import os
import logging
logging.basicConfig(level = logging.DEBUG)
	
num_sample_per_client_training = 2500
num_sample_test = 5000

epochs = 5
batch_size = 32


cfg = [32, 'M', 64, 'M', 128, 'M', 256, 256, 'M', 256, 256, 'M']


class MyModel(nn.Module):
	def __init__(self):
		super(MyModel, self).__init__()
		self.features = self._make_layers(cfg)
		self.classifier = nn.Sequential(
			nn.Linear(256, 128),
			nn.ReLU(True),
			nn.Linear(128, 64),
			nn.ReLU(True),
			nn.Linear(64, 43)
		)

	def forward(self, x):
		out = self.features(x)
		out = out.view(out.size(0), -1)
		out = self.classifier(out)
		output = F.log_softmax(out, dim=1)
		return output

	def _make_layers(self, cfg):
		layers = []
		in_channels = 3
		for x in cfg:
			if x == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
									nn.BatchNorm2d(x),
									nn.ReLU(inplace=True)]
				in_channels = x
		layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
		return nn.Sequential(*layers)


def client_update(client_model, optimizer, train_loader, epoch=5):
	logging.debug("Training avviato.")
	if len(train_loader) == 0:
		logging.error("Il train loader è vuoto. Verifica il dataset.")
	client_model.train()
	for e in range(epoch):
		for batch_idx, (data, target) in enumerate(train_loader):
			logging.debug(f"Processing batch {batch_idx}")
			if torch.cuda.is_available():
				data, target = data.cuda(), target.cuda()
				logging.debug(f"Data shape: {data.shape}, Target shape: {target.shape}")
				#logging.debug("Sto usando CUDA.")
			else:
				data, target = data.to('cpu'), target.to('cpu')
				logging.debug(f"Data shape: {data.shape}, Target shape: {target.shape}")
				#logging.debug("CUDA non disponibile. Sto usando la CPU.")
			optimizer.zero_grad()
			output = client_model(data)
			loss = F.nll_loss(output, target)
			if not torch.isfinite(loss):
				logging.error("Loss non finita: controlla i dati e il modello.")
			logging.debug(f"Loss: {loss.item()}")
			loss.backward()
			logging.debug(f"Loss: {loss.item()}")
			optimizer.step()
			logging.debug(f"Epoch {e+1}, Loss: {loss.item()}")
	logging.debug("Training completato.")
	return loss.item()


def test(model, test_loader):
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			if torch.cuda.is_available():
				data, target = data.cuda(), target.cuda()
			else:
				data, target = data.to('cpu'), target.to('cpu')
			output = model(data)
			test_loss += F.nll_loss(output, target, reduction='sum').item()
			pred = output.argmax(dim=1, keepdim=True)
			correct += pred.eq(target.view_as(pred)).sum().item()

	test_loss /= len(test_loader.dataset)
	acc = correct / len(test_loader.dataset)

	return test_loss, acc


def server_aggregate(global_model, client_models):
	global_dict = global_model.state_dict()
	for k in global_dict.keys():
		global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
	global_model.load_state_dict(global_dict)


def create_train_loader():
	transform_train = transforms.Compose([
		transforms.Resize((32, 32)),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)),
	])

	logging.debug("Scaricamento dataset GTSRB.");

	train_data = datasets.GTSRB('/app/data', split='train', download=True, transform=transform_train)

	logging.debug("Dataset scaricato, sto dividendo randomicamente i batch")

	train_data_split = torch.utils.data.random_split(train_data, [num_sample_per_client_training, len(train_data) - num_sample_per_client_training])[0]
	train_loader = torch.utils.data.DataLoader(train_data_split, batch_size=batch_size, shuffle=True)
	logging.debug("Train loader creato.")
	logging.debug('Dimensioni train loader: %d' % len(train_loader))
	return train_loader


def create_test_loader():
	transform_test = transforms.Compose([
		transforms.Resize((32, 32)),
		transforms.ToTensor(),
		transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)),
	])

	test_data = datasets.GTSRB('/app/data', split='test', download=True, transform=transform_test)

	test_data_split = torch.utils.data.random_split(test_data, [num_sample_test, len(test_data) - num_sample_test])[0]
	test_loader = torch.utils.data.DataLoader(test_data_split, batch_size=batch_size, shuffle=True)
	#print(test_data_split.indices)
	logging.debug("Test loader creato.")
	logging.debug('Dimensioni test loader: %d' % len(test_loader))
	return test_loader
	
`

if (!fs.existsSync(path_dir_models)){
	fs.mkdirSync(path_dir_models)
}

const my_ip = get_ip_addr()

const createNode = async () => {
	const node = await createLibp2p({
		addresses: {
			listen: ['/ip4/' + my_ip + '/tcp/4000']
		},
		transports: [tcp()],
		streamMuxers: [mplex()],
		connectionEncryption: [noise()],
		peerDiscovery: [mdns()]
	})

	return node
}

const node = await createNode()
console.log('MY ADDRESS: ', node.getMultiaddrs(), '\n')


if (process.argv.length >= 3){//worker
	console.log("MASTER_ADDR ricevuto dal worker:", process.argv[2]);
	const master_multiadd = process.argv[2]?.trim().replace(/^\/\//, '/');
	
	master_id = get_peerid_from_multiadd(master_multiadd)

	console.log("Master address:", master_multiadd)
	console.log("Master ID:", master_id)

	await python.ex`
	logging.debug("Creazione train loader.")
	train_loader = create_train_loader()
	if torch.cuda.is_available():
		logging.debug("CUDA disponibile e in uso")
		local_model = MyModel().cuda()
	else:
		logging.debug("CUDA non disponibile; utilizzo la CPU.")
		local_model = MyModel().to('cpu')
	opt = optim.SGD(local_model.parameters(), lr=0.1)
	`
	node.handle('/on_model_received_worker', on_model_received_worker)

}
else{//master
	const NUM_WORKERS = 8
	const NUM_ROUNDS = 150

	//console.log("Attendo creazione nodi worker.")
	//await delay(30000)

	node.handle('/on_model_received_master', on_model_received_master)

	console.log("Avvio peer discovery.")
	node.addEventListener('peer:discovery', async(evt) => {
		console.log("Sono nel codice della peer discovery.")
		console.log("Multiaddr trovati: " + evt.detail.multiaddrs.length + "\n")
		for(let i=0; i < evt.detail.multiaddrs.length; i++){
			console.log("Evento discovery, iterazione n. " + i)
			console.log("Dettagli multiaddr trovato: ", evt.detail.multiaddrs[i].toString())
			console.log("Dettagli Peer ID:", evt.detail.id.toString())
			if(evt.detail.multiaddrs[i].toString().includes('tcp')){
				console.log("Peer trovato.")
				//let peerid = get_peerid_from_multiadd(evt.detail.multiaddrs[i].toString())
				let peerid = evt.detail.id.toString()
				if(peer_id_known_peers.includes(peerid) == false){
					console.log("Aggiungo il peer.")
					peer_id_known_peers.push(peerid)
				}
			}
		}
	})
	console.log("Event listener registrato.")
	
	await python.ex`
	test_loader = create_test_loader()
	if torch.cuda.is_available():
		logging.debug("CUDA disponibile e in uso")
		worker_models = [ MyModel().cuda() for _ in range(${NUM_WORKERS})]
		master_model = MyModel().cuda()
	else:
		logging.debug("CUDA non disponibile. Utilizzo la CPU")
		worker_models = [ MyModel().to('cpu') for _ in range(${NUM_WORKERS})]
		master_model = MyModel().to('cpu')
	`
	console.log("Dataset scaricato, attendo i peer.")
	while(peer_id_known_peers.length < NUM_WORKERS){
		console.log("Sono nel ciclo di waiting su ", peer_id_known_peers.length, " peer.")
		await delay(1000)
	}
	console.log('Peer scoperti, attendo 30 secondi.')
	
	await delay(30000)
	
	console.log('Waiting terminato.')
	
	for (let i = 0; i < NUM_ROUNDS; i++){
		await python.ex`
		torch.save(master_model.state_dict(), ${path_dir_models} + 'master_model')
		`
		const content_model_file = await fs.readFileSync(path_dir_models + 'master_model')
		fs.unlinkSync(path_dir_models + 'master_model')
		
		for(let i = 0; i < peer_id_known_peers.length; i++){
			const stream_to_worker = await node.dialProtocol(peerIdFromString(peer_id_known_peers[i]), '/on_model_received_worker')
			
			await pipe([content_model_file], stream_to_worker)
		}
		
		console.log('sha256 del file modello inviato: ', sha256(content_model_file))
		
		while(num_of_models_received != NUM_WORKERS){
			await delay(1000)
		}

		await python.ex`
		list_files_in_dir = os.listdir(${path_dir_models})
		for i in range(len(worker_models)):
			worker_models[i].load_state_dict(torch.load(${path_dir_models} + list_files_in_dir[i]))

		server_aggregate(master_model, worker_models)
		test_loss, acc = test(master_model, test_loader)
		print('%d-th round, test acc: %0.5f' % (${i}, acc))
		logging.debug('%d-th round, test acc: %0.5f' % (${i}, acc))
		
		for name_file in list_files_in_dir:
			os.remove(${path_dir_models} + name_file)
		`
		lock.acquire('num_of_models_received', function(){
			num_of_models_received = 0
		})
	}
	
}