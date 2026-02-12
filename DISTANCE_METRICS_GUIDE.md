# Guide des Métriques de Distance

## Vue d'ensemble

Votre pipeline dispose maintenant de **5 métriques de distance** pour mesurer les différences entre distributions :

| Métrique | Type | Vitesse | Avantages | Inconvénients |
|----------|------|---------|-----------|---------------|
| **MMD** | Kernel-based | ⚡⚡⚡ Rapide | Flexible, bien testé, pas de suppositions | Nécessite tuning du kernel |
| **FID** | Gaussien | ⚡⚡⚡ Rapide | Standard pour images | Suppose distributions gaussiennes |
| **Sinkhorn** | Optimal Transport | ⚡⚡ Moyen | Théoriquement fondé, métrique | Paramètre de régularisation |
| **Energy** | Distance-based | ⚡⚡ Moyen | Métrique, pas de paramètres | Coûteux en calcul |
| **Adversarial** | Apprentissage | ⚡ Lent | Très puissant, adaptatif | Lent, variance élevée |

## Configuration

### Dans `config.py`

```python
"distance_metrics": [
    # MMD (recommandé par défaut)
    {"name": "mmd", "kernel": "rbf", "gamma": None},  # Auto-tuning

    # Sinkhorn (optimal transport)
    {"name": "sinkhorn", "reg": 0.1, "max_iter": 100},

    # Energy distance (robuste)
    {"name": "energy", "sample_size": 1000},

    # Adversarial (expérimental)
    {"name": "adversarial", "n_critics": 5, "max_iter": 100},
],
```

## Détails des Métriques

### 1. MMD (Maximum Mean Discrepancy)
**Quand l'utiliser** : Par défaut, bon compromis vitesse/qualité.

```python
{"name": "mmd", "kernel": "rbf", "gamma": None}
```

**Paramètres** :
- `kernel`: Type de noyau (`"rbf"` ou `"linear"`)
- `gamma`: Largeur de bande du noyau RBF (None = auto via heuristique de médiane)

**Avantages** :
- ✅ Rapide (pré-calcul de la matrice de kernel)
- ✅ Fonctionne bien avec la normalisation des features
- ✅ Pas de suppositions sur la forme des distributions

**Limitations** :
- ⚠️ Sensible au choix du kernel (corrigé par auto-tuning)

### 2. Sinkhorn Distance (Optimal Transport Régularisé)
**Quand l'utiliser** : Quand vous voulez une vraie métrique avec fondement théorique.

```python
{"name": "sinkhorn", "reg": 0.1, "max_iter": 100}
```

**Paramètres** :
- `reg`: Régularisation entropique (0.01-1.0)
  - Plus petit = plus proche du vrai OT, mais plus lent
  - Plus grand = plus rapide, mais plus régularisé
- `max_iter`: Nombre max d'itérations Sinkhorn-Knopp

**Avantages** :
- ✅ Vraie métrique (satisfait l'inégalité triangulaire)
- ✅ Approxime la distance de Wasserstein
- ✅ Interprétation géométrique claire

**Limitations** :
- ⚠️ Nécessite tuning du paramètre `reg`
- ⚠️ Plus lent que MMD

### 3. Energy Distance
**Quand l'utiliser** : Quand vous voulez une métrique sans paramètres à tuner.

```python
{"name": "energy", "sample_size": 1000}
```

**Paramètres** :
- `sample_size`: Nombre max d'échantillons pour le calcul (pour la vitesse)

**Formule** :
```
E(X, Y) = 2·E[||X - Y||] - E[||X - X'||] - E[||Y - Y'||]
```

**Avantages** :
- ✅ Vraie métrique
- ✅ Pas de paramètres à tuner (sauf sample_size)
- ✅ Robuste

**Limitations** :
- ⚠️ Coût O(n²) en distances (mais échantillonage possible)

### 4. Adversarial Distance (GAN-style Critic)
**Quand l'utiliser** : Pour des expériences avancées, quand la qualité prime sur la vitesse.

```python
{"name": "adversarial", "n_critics": 5, "max_iter": 100}
```

**Paramètres** :
- `n_critics`: Nombre de critics à entraîner (ensemble pour stabilité)
- `max_iter`: Nombre d'itérations d'entraînement par critic

**Principe** :
Entraîne un réseau de neurones (critic) à distinguer les deux distributions.
Approxime la distance de Wasserstein via l'approche WGAN.

**Avantages** :
- ✅ Très puissant (apprend les différences importantes)
- ✅ Adaptatif aux données

**Limitations** :
- ⚠️ Très lent (entraînement de réseaux)
- ⚠️ Variance élevée (nécessite ensemble de critics)
- ⚠️ Expérimental

## Recommandations d'Usage

### Pour des tests rapides
```python
"distance_metrics": [
    {"name": "mmd", "kernel": "rbf", "gamma": None},
]
```

### Pour une évaluation complète
```python
"distance_metrics": [
    {"name": "mmd", "kernel": "rbf", "gamma": None},
    {"name": "sinkhorn", "reg": 0.1, "max_iter": 100},
    {"name": "energy", "sample_size": 1000},
]
```

### Pour comparaison MMD vs Sinkhorn
```python
"distance_metrics": [
    {"name": "mmd", "kernel": "rbf", "gamma": None},
    {"name": "mmd", "kernel": "linear", "gamma": None},
    {"name": "sinkhorn", "reg": 0.01, "max_iter": 200},  # Faible reg
    {"name": "sinkhorn", "reg": 0.5, "max_iter": 50},    # Haute reg
]
```

## Performances Attendues

Sur 100 images de référence + 20 d'évaluation, temps approximatifs :

- **MMD** : ~0.5 sec par configuration
- **FID** : ~0.3 sec par configuration
- **Sinkhorn** : ~2-5 sec par configuration
- **Energy** : ~1-3 sec par configuration
- **Adversarial** : ~30-60 sec par configuration

## Interprétation des Scores

### Scores de Monotonie (Spearman)
- **0.0-0.3** : Mauvaise détection (pas monotone)
- **0.4-0.6** : Détection modérée
- **0.7-0.9** : Bonne détection
- **0.9-1.0** : Excellente détection monotone

### Valeurs Absolues
Les valeurs absolues de distance varient selon la métrique :
- **MMD** : Ordre de grandeur 0.001-0.1
- **Sinkhorn** : Ordre de grandeur 10-1000 (dépend de reg)
- **Energy** : Ordre de grandeur 1-100
- **Adversarial** : Ordre de grandeur 0.01-1.0

**Important** : Ce qui compte c'est la **monotonie** (corrélation de Spearman), pas la valeur absolue !

## Dépannage

### Sinkhorn donne des valeurs très grandes
➡️ Augmentez `reg` (ex: 0.5 ou 1.0) pour plus de régularisation

### Energy Distance négative
➡️ Normal si distributions sont très proches, prenez la valeur absolue

### Adversarial Distance instable
➡️ Augmentez `n_critics` (ex: 10) pour plus de stabilité

### MMD donne toujours les mêmes scores
➡️ Vérifiez que la normalisation des features est activée (devrait être fait automatiquement)

## Tests Unitaires

Testez vos métriques :
```bash
python3 test_new_metrics.py
```

Vous devriez voir :
```
✓ Sinkhorn correctly distinguishes distributions
✓ Energy distance correctly distinguishes distributions
✓ Adversarial distance correctly distinguishes distributions
```
