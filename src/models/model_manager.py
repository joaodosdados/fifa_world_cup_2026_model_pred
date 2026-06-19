"""
Model Manager - Sistema de gerenciamento e seleção de modelos
Gerencia múltiplos modelos treinados e seleciona o melhor automaticamente
"""

import pickle
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime
from sklearn.exceptions import InconsistentVersionWarning

logger = logging.getLogger(__name__)


class ModelManager:
    """Gerencia modelos treinados e suas métricas"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.metadata_file = self.models_dir / "models_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Carrega metadados dos modelos salvos"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Salva metadados dos modelos"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def save_model(self, model: Any, model_name: str, metrics: Dict[str, float], 
                   model_type: str = "sklearn", description: str = ""):
        """
        Salva um modelo com seus metadados
        
        Args:
            model: Modelo treinado
            model_name: Nome único do modelo
            metrics: Dicionário com métricas (accuracy, cv_score, etc)
            model_type: Tipo do modelo (sklearn, ensemble, custom)
            description: Descrição opcional
        """
        # Salvar modelo
        model_path = self.models_dir / f"{model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Salvar metadados
        self.metadata[model_name] = {
            'type': model_type,
            'metrics': metrics,
            'description': description,
            'path': str(model_path),
            'saved_at': datetime.now().isoformat(),
            'active': False
        }
        self._save_metadata()
        
        return model_path
    
    def load_model(self, model_name: str) -> Optional[Any]:
        """Carrega um modelo específico"""
        if model_name not in self.metadata:
            return None
        
        configured_path = Path(self.metadata[model_name]['path'])
        model_path = (
            configured_path
            if configured_path.exists()
            else self.models_dir / f"{model_name}.pkl"
        )
        if not model_path.exists():
            return None
        
        with open(model_path, 'rb') as f:
            # Every loaded estimator is exercised by model_catalog before being
            # exposed. Suppress noisy pickle-version warnings and reject models
            # that fail the runtime prediction check.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", InconsistentVersionWarning)
                return pickle.load(f)

    def get_available_models(self) -> Dict[str, Any]:
        """Load every model whose artifact is currently available."""
        available = {}
        for name in self.metadata:
            try:
                model = self.load_model(name)
            except Exception as exc:
                logger.warning("Skipping model %s: %s", name, exc)
                continue
            if model is not None:
                available[name] = model
        return available
    
    def get_best_model(self, metric: str = 'accuracy') -> tuple[Optional[Any], Optional[str]]:
        """
        Retorna o melhor modelo baseado em uma métrica
        
        Args:
            metric: Métrica para comparação (accuracy, cv_score, etc)
            
        Returns:
            Tupla (modelo, nome_do_modelo)
        """
        if not self.metadata:
            return None, None
        
        # Encontrar modelo com melhor métrica
        best_name = None
        best_score = -1
        
        for name, info in self.metadata.items():
            if metric in info['metrics']:
                score = info['metrics'][metric]
                if score > best_score:
                    best_score = score
                    best_name = name
        
        if best_name:
            return self.load_model(best_name), best_name
        
        return None, None
    
    def set_active_model(self, model_name: str):
        """Define qual modelo está ativo no dashboard"""
        # Desativar todos
        for name in self.metadata:
            self.metadata[name]['active'] = False
        
        # Ativar o selecionado
        if model_name in self.metadata:
            self.metadata[model_name]['active'] = True
            self._save_metadata()
    
    def get_active_model(self) -> tuple[Optional[Any], Optional[str]]:
        """Retorna o modelo atualmente ativo"""
        for name, info in self.metadata.items():
            if info.get('active', False):
                return self.load_model(name), name
        
        # Se nenhum ativo, retorna o melhor
        return self.get_best_model()
    
    def list_models(self) -> List[Dict]:
        """Lista todos os modelos disponíveis com suas métricas"""
        models_list = []
        for name, info in self.metadata.items():
            models_list.append({
                'name': name,
                'type': info['type'],
                'metrics': info['metrics'],
                'description': info['description'],
                'saved_at': info['saved_at'],
                'active': info.get('active', False)
            })
        
        # Ordenar por acurácia
        models_list.sort(key=lambda x: x['metrics'].get('accuracy', 0), reverse=True)
        return models_list
    
    def compare_models(self) -> pd.DataFrame:
        """Retorna DataFrame comparando todos os modelos"""
        if not self.metadata:
            return pd.DataFrame()
        
        data = []
        for name, info in self.metadata.items():
            row = {
                'Model': name,
                'Type': info['type'],
                'Active': '✓' if info.get('active', False) else '',
                **info['metrics']
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        if 'accuracy' in df.columns:
            df = df.sort_values('accuracy', ascending=False)
        
        return df
    
    def delete_model(self, model_name: str):
        """Remove um modelo e seus metadados"""
        if model_name in self.metadata:
            # Deletar arquivo
            model_path = Path(self.metadata[model_name]['path'])
            if model_path.exists():
                model_path.unlink()
            
            # Remover metadados
            del self.metadata[model_name]
            self._save_metadata()
