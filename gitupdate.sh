#!/bin/bash
echo "--- Iniciando atualização ---"
git add .
git commit -m "Atualizacao automatica via script"
git push
echo "--- Sucesso! ---"
