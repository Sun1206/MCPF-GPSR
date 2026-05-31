import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import numpy as np
import os
import traceback
from .blocks.RetrievalStore import *
from .blocks.utils import load_pretrained_stgnn
from .blocks.MLP import MultiLayerPerceptron
from collections import deque

class RAST(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.num_nodes = model_args['num_nodes'] 
        self.input_dim = model_args['input_dim']
        self.output_dim = model_args['output_dim']
        self.embed_dim = model_args.get('embed_dim', 128)
        self.query_dim = model_args.get('query_dim', 32)
        self.retrieval_dim = model_args.get('retrieval_dim', 64)
        self.temporal_dim = model_args.get('temporal_dim', 64)
        self.spatial_dim = model_args.get('spatial_dim', 64)
        self.seq_len = model_args['input_len']
        self.horizon = model_args['output_len']
        self.encoder_layers = model_args.get('encoder_layers', 1)
        self.top_k = model_args.get('top_k', 3)
        self.dropout = model_args.get('dropout', 0.1)
        self.batch_size = model_args.get('batch_size', 32)
        self.add_query = True

        self.update_epoch = 0
        self.update_interval = model_args.get('update_interval', 5)
        self.output_type = 'full'
        print(f"Output type: {self.output_type}")

        self.timing_mode = model_args.get('timing_mode', False)
        self.use_amp = model_args.get('use_amp', False)
        
        self.pre_train_model_name = model_args.get('pre_train_model_name', '')
        self.pre_train_path = model_args.get('pre_train_path', None)
        self.database_path = model_args.get('database_path', './database')

        if self.pre_train_path is not None and self.pre_train_model_name != '':
            try:
                self.backbone = load_pretrained_stgnn(
                    pretrain_path=self.pre_train_path, 
                    model_name=self.pre_train_model_name,
                )
                self.backbone.eval()
                for param in self.backbone.parameters():
                    param.requires_grad = False
                print(f"Loaded pre-trained {self.pre_train_model_name} from {self.pre_train_path}")
            except Exception as e:
                print(f"Error loading pre-trained model: {e}")
                print("Using default backbone")
        else:
            self.backbone = None

        self.mlp_predictor = MultiLayerPerceptron(
            input_dim=self.retrieval_dim + self.query_dim,
            output_dim=self.output_dim * self.horizon,
            dropout=self.dropout
        )

        os.makedirs(self.database_path, exist_ok=True)
        print(f"Database path: {self.database_path}")

        self._init_components()
    
    def _init_components(self):        
        # Calculate expected dataset size for optimization
        expected_dataset_size = max(5000, self.num_nodes * 50)
        
        self.retrieval_store = RetrievalStore(
            self.retrieval_dim, 
            doc_dir=self.database_path,
            max_files=10,
            num_nodes=self.num_nodes,
            seq_len=self.seq_len,
            use_gpu=torch.cuda.is_available(),
            expected_size=expected_dataset_size
        )
        
        self.query_retrieval_proj = nn.Linear(self.query_dim, self.retrieval_dim)
        self.retrieval_ouput_proj = nn.Linear(self.retrieval_dim, self.horizon*self.output_dim)

        self.temporal_proj = nn.Linear(self.temporal_dim, self.retrieval_dim)
        self.spatial_proj = nn.Linear(self.spatial_dim, self.retrieval_dim)
        
        nn.init.orthogonal_(self.query_retrieval_proj.weight)
        nn.init.orthogonal_(self.temporal_proj.weight)
        nn.init.orthogonal_(self.spatial_proj.weight)
        nn.init.zeros_(self.query_retrieval_proj.bias)
        nn.init.zeros_(self.temporal_proj.bias)
        nn.init.zeros_(self.spatial_proj.bias)
        
        self.fusion_dim = self.temporal_dim + self.spatial_dim  

        self.spatial_encoder = nn.Parameter(torch.empty(self.num_nodes, self.spatial_dim))
        nn.init.xavier_uniform_(self.spatial_encoder)
        
        self.temporal_encoder = nn.Conv2d(
            in_channels=self.input_dim * self.seq_len,
            out_channels=self.temporal_dim,
            kernel_size=(1, 1),
            bias=True
        )
        nn.init.kaiming_normal_(self.temporal_encoder.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.temporal_encoder.bias)

        self.feature_encoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.fusion_dim, self.fusion_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ) for _ in range(self.encoder_layers)
        ])
        
        for layer in self.feature_encoder_layers:
            for m in layer.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        self.horizon_layer = nn.Linear(self.fusion_dim, self.horizon)
        nn.init.xavier_normal_(self.horizon_layer.weight)
        nn.init.zeros_(self.horizon_layer.bias)

        self.attn = nn.MultiheadAttention(num_heads=4, embed_dim=self.retrieval_dim, dropout=0.1)
        self.output_projector = nn.Linear(self.output_dim, self.output_dim)
        nn.init.xavier_normal_(self.output_projector.weight)
        nn.init.zeros_(self.output_projector.bias)
        
        self.hidden_to_query_proj = nn.Linear(self.fusion_dim, self.query_dim)
        nn.init.xavier_normal_(self.hidden_to_query_proj.weight)
        nn.init.zeros_(self.hidden_to_query_proj.bias)
        
        # Retrieval metrics tracking
        self.retrieval_metrics = {
            'avg_gate_weight': 0.0,
            'avg_similarity': 0.0,
            'avg_quality_score': 0.0,
            'update_count': 0
        }

    @torch.no_grad()
    def _update_retrieval_tensors(self, temp_embed: torch.Tensor, node_embed: torch.Tensor, 
                                 history_data: torch.Tensor, epoch: int):
        epoch_id = epoch % 50
        max_tensors = 20
        max_vectors = 1000

        try:
            B, temporal_dim, N, _ = temp_embed.shape
            B, spatial_dim, N, _ = node_embed.shape
            
            temp_processed = temp_embed.squeeze(-1)
            node_processed = node_embed.squeeze(-1)
            
            temp_4d = temp_processed.unsqueeze(-1).expand(-1, -1, -1, epoch_id + 1)
            node_4d = node_processed.unsqueeze(-1).expand(-1, -1, -1, epoch_id + 1)
            
            temp_flat = temp_processed.reshape(-1, temporal_dim)
            node_flat = node_processed.reshape(-1, spatial_dim)
            
            temp_proj = self.temporal_proj(temp_flat)
            node_proj = self.spatial_proj(node_flat)
            
            if not hasattr(self.retrieval_store, 'temporal_tensors'):
                self.retrieval_store.temporal_tensors = []
                self.retrieval_store.spatial_tensors = []
            
            temp_tensor_np = temp_4d.detach().cpu().numpy().astype(np.float32)
            node_tensor_np = node_4d.detach().cpu().numpy().astype(np.float32)
            temp_proj_np = temp_proj.detach().cpu().numpy().astype(np.float32)
            node_proj_np = node_proj.detach().cpu().numpy().astype(np.float32)
            
            self.retrieval_store.temporal_tensors.append(temp_tensor_np)
            self.retrieval_store.spatial_tensors.append(node_tensor_np)
            
            for vec in temp_proj_np:
                self.retrieval_store.temporal_vectors.append(vec)
            for vec in node_proj_np:
                self.retrieval_store.spatial_vectors.append(vec)
            
            if len(self.retrieval_store.temporal_tensors) > max_tensors:
                self.retrieval_store.temporal_tensors = self.retrieval_store.temporal_tensors[-max_tensors:]
                self.retrieval_store.spatial_tensors = self.retrieval_store.spatial_tensors[-max_tensors:]
            
            # Convert deque to list, slice, then recreate deque with same maxlen
            if len(self.retrieval_store.temporal_vectors) > max_vectors:
                temp_list = list(self.retrieval_store.temporal_vectors)
                maxlen = self.retrieval_store.temporal_vectors.maxlen
                self.retrieval_store.temporal_vectors = deque(temp_list[-max_vectors:], maxlen=maxlen)
            if len(self.retrieval_store.spatial_vectors) > max_vectors:
                spat_list = list(self.retrieval_store.spatial_vectors)
                maxlen = self.retrieval_store.spatial_vectors.maxlen
                self.retrieval_store.spatial_vectors = deque(spat_list[-max_vectors:], maxlen=maxlen)

            self.retrieval_store._rebuild_indices()
            
        except Exception as e:
            print(f"Error updating retrieval tensors: {e}")
            traceback.print_exc()

    def _retrieve_tensors(self, query: torch.Tensor, history_data: torch.Tensor, 
                         embed: torch.Tensor, temporal: bool = True) -> torch.Tensor:
        B, N, E = query.shape
        device = query.device
        
        if temporal:
            vectors = self.retrieval_store.temporal_vectors
            tensors = getattr(self.retrieval_store, 'temporal_tensors', [])
            proj_layer = self.temporal_proj
        else:
            vectors = self.retrieval_store.spatial_vectors
            tensors = getattr(self.retrieval_store, 'spatial_tensors', [])
            proj_layer = self.spatial_proj
        
        if not vectors or not tensors:
            return torch.zeros(B, N, self.retrieval_dim, device=device, dtype=query.dtype)
        
        query_flat = query.reshape(-1, E)
        query_projected = self.query_retrieval_proj(query_flat)
        query_np = query_projected.detach().cpu().numpy().astype(np.float32)
        
        if not query_np.flags['C_CONTIGUOUS']:
            query_np = np.ascontiguousarray(query_np)
        
        if query_np.ndim != 2:
            raise ValueError(f"FAISS expects 2D array, got {query_np.ndim}D array with shape {query_np.shape}")
        
        k_limit = min(self.top_k, 3)
        distances, indices = self.retrieval_store.search(query_np, k=k_limit, temporal=temporal)
        
        if len(tensors) > 0:
            tensor_sample = tensors[0]
            num_vectors_per_tensor = tensor_sample.shape[0] * tensor_sample.shape[2]
            
            tensor_indices = indices.flatten() // max(1, num_vectors_per_tensor)
            tensor_indices = np.clip(tensor_indices, 0, len(tensors)-1)
            unique_tensor_indices = np.unique(tensor_indices)[:k_limit]
            
            retrieved_tensors = []
            for tensor_idx in unique_tensor_indices:
                tensor_4d = tensors[tensor_idx]
                tensor_torch = torch.from_numpy(tensor_4d).to(device, dtype=query.dtype)
                
                tensor_avg = torch.mean(tensor_torch, dim=-1)
                tensor_flat = tensor_avg.permute(0, 2, 1).reshape(-1, tensor_avg.shape[1])
                tensor_proj = proj_layer(tensor_flat)
                tensor_reshaped = tensor_proj.reshape(tensor_avg.shape[0], tensor_avg.shape[2], -1)
                retrieved_tensors.append(tensor_reshaped)
            
            if retrieved_tensors:
                stacked_tensors = torch.stack(retrieved_tensors, dim=0)
                averaged_tensor = torch.mean(stacked_tensors, dim=0)
                return averaged_tensor
        
        return torch.zeros(B, N, self.retrieval_dim, device=device, dtype=query.dtype)

    def temporal_retriever(self, query: torch.Tensor, history_data: torch.Tensor, temp_embed: torch.Tensor) -> torch.Tensor:
        return self._retrieve_tensors(query, history_data, temp_embed, temporal=True)
    
    def spatial_retriever(self, query: torch.Tensor, history_data: torch.Tensor, node_embed: torch.Tensor) -> torch.Tensor:
        return self._retrieve_tensors(query, history_data, node_embed, temporal=False)
    
    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, 
               batch_seen: int, epoch: int, train: bool, **kwargs) -> dict:
        B, L, N, D = history_data.shape
    
        if self.use_amp and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                return self._forward_impl(history_data, future_data, batch_seen, epoch, train)
        else:
            return self._forward_impl(history_data, future_data, batch_seen, epoch, train)
    
    def _forward_impl(self, history_data: torch.Tensor, future_data: torch.Tensor, 
                     batch_seen: int, epoch: int, train: bool) -> dict:
        B, L, N, D = history_data.shape

        input_data = history_data[..., range(self.input_dim)]
        
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(B, N, -1).transpose(1, 2).unsqueeze(-1)
        
        temp_embed = self.temporal_encoder(input_data)
        node_embed = self.spatial_encoder.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1)

        should_update_retrieval = (
            train and self.output_type in ["full", "only_retrieval_embed","without_query_embedding"] and
            epoch % self.update_interval == 0 and self.update_epoch != epoch)
        
        if should_update_retrieval:
            self.update_epoch = epoch
            print(f"Updating retrieval tensors at epoch {epoch}")
            self._update_retrieval_tensors(temp_embed, node_embed, history_data, epoch)

        hidden = torch.cat([temp_embed, node_embed], dim=1)
        feature_fusion = hidden.squeeze(-1).transpose(1, 2)
        
        for layer in self.feature_encoder_layers:
            feature_fusion = layer(feature_fusion) + feature_fusion
        
        query_embed = self.horizon_layer(feature_fusion)

        prediction = query_embed.unsqueeze(-1).expand(-1, -1, -1, self.output_dim)
        prediction = self.output_projector(query_embed.reshape(-1, self.output_dim))
        prediction = prediction.reshape(B, N, self.horizon, self.output_dim)
        prediction = prediction.permute(0, 2, 1, 3)

        if self.output_type == "only_query":
            return {'prediction': prediction}
        
        if self.backbone is not None:
            prediction = self.backbone(history_data, future_data, batch_seen, epoch, train)

        query_embed = self.hidden_to_query_proj(feature_fusion)

        temporal_retrieval = self.temporal_retriever(query_embed, history_data, temp_embed)
        spatial_retrieval = self.spatial_retriever(query_embed, history_data, node_embed)
        
        if self.output_type == "without_temporal_retrieval":
            temporal_retrieval = torch.zeros_like(temporal_retrieval)
        if self.output_type == "without_spatial_retrieval":
            spatial_retrieval = torch.zeros_like(spatial_retrieval)
        if self.output_type == "only_retrieval":
            query_embed = torch.zeros_like(query_embed)
            retrieval_embed = torch.cat([temporal_retrieval, spatial_retrieval], dim=-1)
            output = self.retrieval_ouput_proj(retrieval_embed)
            output = output.view(B, N, self.horizon, self.output_dim).transpose(1, 2)
            return {'prediction': output}

        query_retrieval = self.query_retrieval_proj(query_embed)
        temporal_retrieval = self.attn(query_retrieval, temporal_retrieval, temporal_retrieval)[0]
        spatial_retrieval = self.attn(query_retrieval, spatial_retrieval, spatial_retrieval)[0]
        retrieval_embed = self.attn(query_retrieval, temporal_retrieval, spatial_retrieval)[0]

        if self.output_type == "without_retrieval_embedding":
            retrieval_embed = torch.zeros_like(retrieval_embed)
        elif self.output_type == "without_query_embedding":
            query_embed = torch.zeros_like(query_embed)

        combined_embed = torch.cat([retrieval_embed, query_embed], dim=-1)
        
        output = self.mlp_predictor(combined_embed)
        
        output = output.view(B, N, self.horizon, self.output_dim).transpose(1, 2)
        
        if self.add_query:
            output = output + prediction
        
        # Return standard results
        result = {
            'prediction': output,
            'retrieval_metrics': self.retrieval_metrics.copy()
        }
        
        return result
