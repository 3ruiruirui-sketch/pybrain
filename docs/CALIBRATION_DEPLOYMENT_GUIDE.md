# Guia de Deployment: Calibração Platt (Sem Risco)

**Data:** 2026-04-13  
**Objetivo:** Aplicar coeficientes de calibração em produção de forma segura

---

## 1. Resumo da Infraestrutura

```
┌─────────────────────────────────────────────────────────────────┐
│  FASE 1: GERAÇÃO (Offline)                                      │
│  ─────────────────────────                                      │
│  compute_platt_calibration.py  →  platt_coefficients.json        │
│                                                                 │
│  FASE 2: DEPLOYMENT (Sem risco)                                 │
│  ─────────────────────────────                                  │
│  Copiar JSON → models/calibration/                              │
│  Aplicação automática via apply_platt_calibration()            │
│  Fallback: identity se ficheiro não existir                    │
└─────────────────────────────────────────────────────────────────┘
```

**Ficheiros-chave:**
- `scripts/compute_platt_calibration.py` - Gera coeficientes
- `models/calibration/platt_coefficients.json` - Coeficientes por canal
- `pybrain/config/defaults.yaml` - Configuração do caminho
- `apply_platt_calibration()` - Função de aplicação automática

---

## 2. Passo a Passo de Deployment

### Passo 1: Gerar Coeficientes (Ambiente Seguro)

```bash
cd ~/Downloads/PY-BRAIN
source .venv/bin/activate

# Gerar coeficientes (não afeta produção!)
python scripts/compute_platt_calibration.py \
    --brats_dir data/datasets/BraTS2021/raw/BraTS2021_Training_Data \
    --bundle_dir models/brats_bundle \
    --n_cases 50 \
    --device mps \
    --out_dir models/calibration \
    --skip_swinunetr  # SwinUNETR desativado no teu setup

# Tempo estimado: ~50 casos × 40s = ~30-40 minutos
```

**Output:**
```
models/calibration/
├── platt_coefficients.json      ← ⭐ COEFICIENTES (copiar este)
├── ensemble_weights.json        ← Pesos recomendados
└── per_model_dice.json          ← Log de validação
```

### Passo 2: Validar Coeficientes Gerados

```bash
# Verificar estrutura
cat models/calibration/platt_coefficients.json

# Output esperado:
# {
#   "tc": {"A": 1.15, "B": -0.23},
#   "wt": {"A": 1.08, "B": -0.15},
#   "et": {"A": 0.95, "B": -0.08}
# }

# Validar métricas
python validate_calibration_metrics.py
```

**Critérios de aceitação:**
- ✅ A entre 0.8 e 1.3 (não extremo)
- ✅ B entre -0.5 e 0.5 (bias moderado)
- ✅ Casos usados ≥ 30 (estatisticamente robusto)
- ⚠️ Se A < 0.5 ou A > 2.0 → modelo pode estar muito descalibrado

### Passo 3: Teste de Staging (Sem Impacto em Produção)

```bash
# Criar backup do config
cp pybrain/config/defaults.yaml pybrain/config/defaults.yaml.backup

# NÃO é necessário editar config! O apply_platt_calibration já:
# 1. Procura ficheiro em models/calibration/platt_coefficients.json
# 2. Se existir → aplica calibração
# 3. Se não existir → fallback identity (sem mudança)

# Testar num caso conhecido
export PYBRAIN_SESSION=/Users/ssoares/Downloads/PY-BRAIN/results/debug_session_20260413_020415/session.json

# Backup dos resultados atuais
cp results/debug_session_20260413_020415/segmentation_full.nii.gz \
   results/debug_session_20260413_020415/segmentation_full_backup.nii.gz

# Correr pipeline (agora VAI aplicar calibração se ficheiro existe)
python scripts/3_brain_tumor_analysis.py 2>&1 | tee /tmp/with_calibration.log

# Comparar resultados
python -c "
import nibabel as nib
import numpy as np

seg_old = nib.load('results/debug_session_20260413_020415/segmentation_full_backup.nii.gz').get_fdata()
seg_new = nib.load('results/debug_session_20260413_020415/segmentation_full.nii.gz').get_fdata()

# Dice entre old e new
def dice(a, b):
    return 2.0 * np.sum(a * b) / (np.sum(a) + np.sum(b))

d_wt = dice(seg_old > 0, seg_new > 0)
d_tc = dice(((seg_old==1)|(seg_old==4)), ((seg_new==1)|(seg_new==4)))
d_et = dice(seg_old==4, seg_new==4)

print(f'Agreement Old vs New:')
print(f'  WT: {d_wt:.4f}')
print(f'  TC: {d_tc:.4f}')
print(f'  ET: {d_et:.4f}')
print(f'  Status: {\"✅ OK\" if d_wt > 0.95 else \"⚠️ REVIEW\"}')'
```

### Passo 4: Decisão Go/No-Go

| Métrica | Threshold | Ação |
|---------|-----------|------|
| Dice Old vs New > 0.95 | ✅ PASS | Prosseguir com deploy |
| Dice Old vs New 0.90-0.95 | ⚠️ REVIEW | Inspecionar visualmente |
| Dice Old vs New < 0.90 | ❌ FAIL | Rejeitar calibração |
| ECE (se calculado) < 0.05 | ✅ PASS | Boa calibração |
| ECE > 0.15 | ❌ FAIL | Recalcular com mais casos |

### Passo 5: Deploy em Produção (Se Aprovado)

```bash
# O ficheiro já está no sítio certo (models/calibration/)
# Nenhuma alteração de código necessária!

# Verificar que está a ser usado:
grep "Platt scaling applied" /tmp/with_calibration.log

# Output esperado:
# [INFO] Platt scaling applied from saved coefficients.

# Se ver log:
# [WARNING] Platt calibration coefficients not found.
# → Verificar caminho do ficheiro
```

---

## 3. Configuração do Caminho (Confirmar)

Verificar `pybrain/config/defaults.yaml`:

```yaml
models:
  platt_calibration:
    coefficients_file: "models/calibration/platt_coefficients.json"
```

**Fallback automático:**
- Se ficheiro não existir → aplica identity (sem calibração)
- Se ficheiro inválido → warning + identity
- Se coeficientes incompletos → canais com dados usam Platt, outros usam identity

---

## 4. Validação Contínua

### 4.1 Métricas ECE/Brier (Recomendado)

Adicionar ao `compute_platt_calibration.py` (já implementado):

```python
# Após fitting, calcular ECE
from sklearn.metrics import brier_score_loss

# Para cada canal
for ch in ['tc', 'wt', 'et']:
    p = probs[ch]
    y = ground_truth[ch]
    
    ece, _ = compute_ece(y, p, n_bins=10)
    brier = brier_score_loss(y, p)
    
    print(f"{ch}: ECE={ece:.4f}, Brier={brier:.4f}")
```

**Interpretação:**
- ECE < 0.05: Excelente calibração
- ECE 0.05-0.10: Boa calibração
- ECE > 0.15: Calibração fraca (reconsiderar)
- Brier score: comparar antes/depois (mais baixo = melhor)

### 4.2 Reliability Diagram (Visual)

```python
# Gerar reliability diagram
from sklearn.calibration import calibration_curve

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for idx, ch in enumerate(['tc', 'wt', 'et']):
    prob_true, prob_pred = calibration_curve(y, p, n_bins=10)
    ax[idx].plot(prob_pred, prob_true, 's-')
    ax[idx].plot([0, 1], [0, 1], 'k--')  # Perfect calibration
    ax[idx].set_title(f'{ch.upper()} Calibration')
    ax[idx].set_xlabel('Mean Predicted Probability')
    ax[idx].set_ylabel('Fraction of Positives')
```

---

## 5. Rollback Plan (Se Necessário)

### Opção A: Remover Ficheiro (Mais Seguro)

```bash
# Simplesmente remover o ficheiro → fallback automático para identity
mv models/calibration/platt_coefficients.json \
   models/calibration/platt_coefficients.json.disabled

# Verificar que fallback funciona:
python scripts/3_brain_tumor_analysis.py 2>&1 | grep -i "fallback\|identity"
# [WARNING] Platt calibration coefficients not found. No probability adjustment...
```

### Opção B: Editar Config (Desativar)

```yaml
# pybrain/config/defaults.yaml
models:
  platt_calibration:
    coefficients_file: "models/calibration/platt_coefficients.json.disabled"
    # Ou caminho inválido para forçar fallback
```

### Verificação de Rollback

```bash
# Confirmar que não há calibração
python -c "
import json
from pathlib import Path

path = Path('models/calibration/platt_coefficients.json')
print(f'Ficheiro existe: {path.exists()}')
if path.exists():
    print('Calibração ATIVA')
else:
    print('Calibração INATIVA (fallback identity)')
"
```

---

## 6. Checklist de Deployment

- [ ] Executar `compute_platt_calibration.py` com ≥30 casos
- [ ] Verificar `platt_coefficients.json` tem A, B para tc/wt/et
- [ ] Validar valores de A (0.8-1.3) e B (-0.5 a 0.5)
- [ ] Executar `validate_calibration_metrics.py`
- [ ] Testar num caso conhecido (staging)
- [ ] Comparar Dice antes/depois (> 0.95 aceitável)
- [ ] Verificar log mostra "Platt scaling applied"
- [ ] Confirmar outputs (masks) são equivalentes
- [ ] Documentar métricas ECE/Brier se calculadas
- [ ] Criar rollback plan (backup do JSON)

---

## 7. Resumo de Comandos

```bash
# 1. Gerar coeficientes
python scripts/compute_platt_calibration.py \
    --brats_dir data/datasets/BraTS2021/raw/BraTS2021_Training_Data \
    --n_cases 50 --device mps --skip_swinunetr

# 2. Validar
python validate_calibration_metrics.py

# 3. Testar staging
export PYBRAIN_SESSION=...
python scripts/3_brain_tumor_analysis.py

# 4. Verificar aplicação
grep "Platt scaling applied" /tmp/pipeline.log

# 5. Rollback (se necessário)
mv models/calibration/platt_coefficients.json{,.disabled}
```

---

## 8. Risco Assessment

| Risco | Probabilidade | Mitigação |
|-------|---------------|-----------|
| Calibração piora resultados | Baixa | Teste staging com Dice comparison |
| Coeficientes inválidos | Baixa | Validação A/B ranges |
| Ficheiro corrompido | Baixa | JSON schema validation no load |
| Performance degrade | Muito baixa | Overhead computacional mínimo |
| Impossível rollback | Nula | Apenas remover ficheiro |

**Veredito:** ✅ **Risco MUITO BAIXO** — fallback automático, teste fácil, rollback trivial.

---

## Nota Final

O teu PY-BRAIN já tem **toda a infraestrutura implementada**. Não precisas de alterar código, apenas:
1. Correr o script de calibração (gera coeficientes)
2. Verificar qualidade dos coeficientes
3. Testar num caso (staging)
4. Se aprovado → ficheiro já está no sítio certo para produção!

**Próximo passo recomendado:** Executar `compute_platt_calibration.py` com 50 casos BraTS.
