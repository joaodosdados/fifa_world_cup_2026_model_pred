#!/usr/bin/env python3
"""
Script para limpar completamente o cache do Streamlit
"""
import os
import shutil
from pathlib import Path

# Diretórios de cache do Streamlit
cache_dirs = [
    Path.home() / ".streamlit" / "cache",
    Path(".streamlit") / "cache",
    Path("__pycache__"),
]

print("🧹 Limpando cache do Streamlit...")
for cache_dir in cache_dirs:
    if cache_dir.exists():
        try:
            shutil.rmtree(cache_dir)
            print(f"✓ Removido: {cache_dir}")
        except Exception as e:
            print(f"✗ Erro ao remover {cache_dir}: {e}")

print("\n✅ Cache limpo! Agora execute:")
print("   streamlit run main.py")

# Made with Bob
