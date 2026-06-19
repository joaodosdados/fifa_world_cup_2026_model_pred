#!/bin/bash
# Script para subir o Streamlit na EC2 remotamente
# Uso: ./deploy_streamlit.sh [retreinar]
#   sem argumento -> só sobe o streamlit
#   "retreinar"   -> roda o train_models.py antes de subir

KEY="fifaWorldCup.pem"
HOST="ec2-user@52.91.250.59"
PROJECT_DIR="fifa_world_cup_2026_model_pred"
RETREINAR="$1"

if [ "$RETREINAR" == "retreinar" ]; then
  echo ">> Retreinando modelos e subindo o Streamlit..."
  ssh -i "$KEY" "$HOST" "
    cd $PROJECT_DIR &&
    source .venv/bin/activate &&
    pkill -f streamlit;
    python scripts/train_models.py &&
    nohup streamlit run main.py --server.port 8501 --server.address 0.0.0.0 > streamlit.log 2>&1 &
    sleep 2 &&
    cat streamlit.log
  "
else
  echo ">> Subindo o Streamlit..."
  ssh -i "$KEY" "$HOST" "
    cd $PROJECT_DIR &&
    source .venv/bin/activate &&
    pkill -f streamlit;
    nohup streamlit run main.py --server.port 8501 --server.address 0.0.0.0 > streamlit.log 2>&1 &
    sleep 2 &&
    cat streamlit.log
  "
fi

echo ""
echo ">> Acesse: http://52.91.250.59:8501"