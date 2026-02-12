# Guide d'Utilisation - Lancer des Expérimentations

## Vue d'ensemble

Ce guide explique comment configurer et lancer des expérimentations avec le pipeline Métrique pour évaluer la qualité d'images en utilisant différentes couches de réseaux de neurones pré-entraînés.

**Pipeline en 7 étapes:**
1. Charger N_R images de référence (VisDrone2019)
2. Extraire features d'un backbone à une couche spécifique
3. Calculer matrices de Gram → Distribution D_R
4. Charger N_E images d'évaluation (différentes)
5. Appliquer dégradations progressives → Distributions D_E_k, calculer distance MMD avec D_R
6. Évaluer monotonie avec corrélation de Spearman
7. Sauvegarder résultats en CSV

---

## 1. Sélectionner les Couches (Layer Selection)

### Méthode Simple

1. **Ouvrir** `config.py`
2. **Trouver** la section `ENABLED_LAYERS` (ligne ~312)
3. **Modifier** les valeurs `True`/`False` pour activer/désactiver des couches
4. **Sauvegarder** le fichier

### Exemple: Activer seulement les couches tardives (late layers)

```python
ENABLED_LAYERS = {
    "sd_vae": {
        0: False,  # Désactiver early layer
        3: False,  # Désactiver early-mid
        6: False,  # Désactiver mid
        9: True,   # Activer mid-late ✓
        12: True,  # Activer late ✓
        16: True,  # Activer final ✓
    },
    # Autres backbones...
}
```

### Exemple: Tester un seul backbone rapidement

```python
ENABLED_LAYERS = {
    "sd_vae": {
        0: True,   # early
        9: True,   # mid-late
        16: True,  # final
    },
    "dinov2_vitb14": {
        # Désactiver TOUTES les couches (set all to False)
        0: False,
        1: False,
        # ... (all False)
    },
    "vgg19": {
        # Désactiver TOUTES les couches
        # ... (all False)
    },
    # Autres backbones tous désactivés...
}
```

### Configuration Actuelle (Recommandée)

Par défaut, ENABLED_LAYERS contient une sélection de **6 couches par backbone** couvrant la hiérarchie complète (early → mid → late). Cette configuration est optimale pour explorer la progression sémantique.

**Total:** ~32 configurations activées par défaut

---

## 2. Lancer une Expérimentation

### Commandes Principales

#### Mode Medium (Recommandé)
```bash
python main.py --mode medium
```
- Utilise MMD/FID uniquement (pas de GMM)
- Rapide et efficace
- Recommandé pour la plupart des cas

#### Mode Full
```bash
python main.py --mode full
```
- Évaluation complète (ancienne version avec GMM)
- Plus lent
- Utilise toutes les configurations

#### Mode Quick Test
```bash
python main.py --mode quick
```
- Test rapide sur quelques couches prédéfinies
- Utile pour vérifier que le pipeline fonctionne

### Options Avancées

#### Spécifier un backbone spécifique
```bash
# Un seul backbone
python main.py --mode medium --backbone sd_vae

# Plusieurs backbones
python main.py --mode medium --backbone sd_vae vgg19

# Tous les principaux backbones (dinov2, vgg19, sd_vae, lpips_vgg)
python main.py --mode medium --backbone all
```

**Note:** Les couches utilisées sont toujours celles définies dans `ENABLED_LAYERS` du config.py

#### Modifier le nombre d'images
```bash
# Changer le nombre d'images de référence
python main.py --mode medium --n-inference 50

# Changer le nombre d'images d'évaluation
python main.py --mode medium --n-evaluation 10

# Les deux ensemble
python main.py --mode medium --n-inference 50 --n-evaluation 10
```

**Rappel:**
- `n_images_inference` (N_R): Images de référence pour la distribution D_R (défaut: 100)
- `n_images_evaluation` (N_E): Images à dégrader (défaut: 20)

---

## 3. Construire le Cache d'Activations (Optionnel)

Pour accélérer les expérimentations répétées, vous pouvez pré-calculer les activations:

```bash
# Cache pour SD VAE, layer 9
python main.py --build-cache --backbone sd_vae --layers 9

# Cache pour plusieurs layers
python main.py --build-cache --backbone sd_vae --layers 0 3 6 9 12 16

# Cache pour tous les backbones activés avec certaines couches
python main.py --build-cache --backbone all --layers 0 9 16
```

**Avantage:** Les activations sont calculées une fois et réutilisées pour tous les metrics/transforms.

**Location:** Cache sauvegardé dans `results/cache/activations/`

---

## 4. Interpréter les Résultats

### Fichiers de Sortie

Les résultats sont sauvegardés dans le dossier `results/` :
- `resultats_medium_YYYYMMDD_HHMMSS.csv` - Résultats du mode medium
- `resultats_YYYYMMDD_HHMMSS.csv` - Résultats du mode full
- `monotonicity_curves.png` - Graphiques de monotonie par dégradation

### Format CSV

| Colonne | Description |
|---------|-------------|
| `backbone` | Modèle utilisé (sd_vae, vgg19, etc.) |
| `layers` | Nom de la configuration de couche (e.g., "single_layer_9") |
| `transform` | Type de transformation (e.g., "gram_spatial") |
| `distribution` | Modèle de distribution ("empirical" pour MMD/FID) |
| `metric` | Métrique de distance (e.g., "mmd[rbf,auto]") |
| `monotonicity` | Dict avec corrélation de Spearman par type de dégradation |
| `mean_monotonicity` | Moyenne des corrélations de Spearman |
| `detailed_scores` | Scores de distance pour chaque niveau de dégradation |

### Interprétation

**Monotonicity (Corrélation de Spearman):**
- **1.0** : Parfaite monotonie (distance augmente strictement avec dégradation)
- **0.9-1.0** : Excellente monotonie
- **0.7-0.9** : Bonne monotonie
- **< 0.7** : Monotonie faible (métrique moins fiable)

**Mean Monotonicity:**
- Moyenne sur tous les types de dégradation (blur, noise, aliasing, blur_contrast)
- Indicateur global de la qualité de la métrique

**Exemple:**
```
backbone: sd_vae
layers: single_layer_9
metric: mmd[rbf,auto]
mean_monotonicity: 0.92
```
→ Layer 9 de SD VAE avec MMD RBF a une excellente monotonie (0.92)

---

## 5. Cas d'Usage Courants

### Cas 1: Comparer Couches Précoces vs Tardives

**Objectif:** Voir si les couches tardives capturent mieux la qualité d'image que les précoces

**Étapes:**
1. Dans `config.py`, activer seulement les couches précoces (0-3) pour SD VAE
2. Lancer: `python main.py --mode medium --backbone sd_vae`
3. Noter le `mean_monotonicity` dans le CSV
4. Modifier `config.py` pour activer seulement les couches tardives (12-16)
5. Re-lancer: `python main.py --mode medium --backbone sd_vae`
6. Comparer les `mean_monotonicity` des deux runs

**Résultat attendu:** Les couches tardives devraient avoir une monotonie légèrement meilleure (plus sémantiques)

---

### Cas 2: Trouver la Meilleure Couche pour un Backbone

**Objectif:** Identifier quelle couche donne la meilleure sensibilité aux dégradations

**Étapes:**
1. Dans `config.py`, activer TOUTES les couches pour un backbone (e.g., SD VAE: 0-16 = True)
2. Lancer: `python main.py --mode medium --backbone sd_vae`
3. Ouvrir le CSV et trier par `mean_monotonicity` décroissant
4. Identifier la couche avec le score le plus élevé

**Analyse:**
- Comparer `mean_monotonicity` entre couches
- Regarder `detailed_scores` pour voir la progression des distances
- Considérer aussi le type de dégradation (certaines couches meilleures pour certaines dégradations)

---

### Cas 3: Comparaison Multi-Backbones

**Objectif:** Comparer quel backbone est le plus sensible aux dégradations

**Étapes:**
1. Dans `config.py`, activer une sélection représentative de couches pour TOUS les backbones
   ```python
   # Par exemple: early, mid, late pour chaque backbone
   sd_vae: [0, 9, 16]
   vgg19: [0, 7, 15]
   dinov2_vitb14: [0, 5, 13]
   lpips_vgg: [0, 7, 15]
   ```
2. Lancer: `python main.py --mode medium --backbone all`
3. Comparer les `mean_monotonicity` entre backbones dans le CSV

**Analyse:**
- Quel backbone a la monotonie la plus élevée en moyenne?
- Y a-t-il des backbones meilleurs pour certains types de dégradations?
- Comparer CNN (VGG, LPIPS) vs ViT (DinoV2) vs VAE (SD VAE)

---

### Cas 4: Test Rapide d'une Configuration

**Objectif:** Vérifier rapidement que le pipeline fonctionne avant un long run

**Étapes:**
1. Dans `config.py`, activer seulement 2-3 couches pour un backbone
   ```python
   sd_vae: [0, 9, 16] → Activer
   Tous les autres → Désactiver
   ```
2. Lancer avec moins d'images:
   ```bash
   python main.py --mode medium --backbone sd_vae --n-inference 20 --n-evaluation 10
   ```
3. Vérifier que le CSV est généré et contient des résultats cohérents

**Temps:** ~5-10 minutes (au lieu de plusieurs heures pour un run complet)

---

### Cas 5: Expérimentation Complète (Overnight)

**Objectif:** Explorer exhaustivement toutes les configurations

**Étapes:**
1. Dans `config.py`, activer TOUTES les couches intéressantes (6-8 par backbone)
2. Garder les paramètres par défaut (n_inference=100, n_evaluation=20)
3. Lancer le soir:
   ```bash
   nohup python main.py --mode medium --backbone all > experiment.log 2>&1 &
   ```
4. Le lendemain matin, analyser les résultats

**Temps estimé:** 6-12 heures (selon GPU/CPU)

**Total configurations:** ~32 (6 layers × 4 backbones × 1 transform × 1 metric)

---

## 6. Conseils et Astuces

### Optimisation de Performance

1. **Utiliser le cache d'activations** pour les layers fréquemment testées:
   ```bash
   python main.py --build-cache --backbone sd_vae --layers 0 3 6 9 12 16
   ```
   Puis lancer les expérimentations normalement (le cache sera utilisé automatiquement)

2. **Réduire le nombre d'images** pour des tests rapides:
   ```bash
   python main.py --mode medium --n-inference 50 --n-evaluation 10
   ```
   (La monotonie devrait rester stable même avec moins d'images)

3. **Désactiver les transformers** si pas nécessaire:
   - DinoV2 et CLIP sont plus lents que VGG/ResNet
   - Désactiver en mettant toutes leurs couches à False

### Debugging

Si le pipeline échoue:

1. **Vérifier les logs**:
   ```
   Lisez les messages d'erreur dans le terminal
   ```

2. **Tester en mode quick**:
   ```bash
   python main.py --mode quick
   ```

3. **Vérifier les couches activées**:
   ```bash
   python -c "from config import get_enabled_experiment_configs; print(len(get_enabled_experiment_configs()), 'configs')"
   ```

4. **Tester avec un petit nombre d'images**:
   ```bash
   python main.py --mode medium --n-inference 10 --n-evaluation 5
   ```

### Bonne Pratiques

1. **Documenter vos configurations**:
   - Avant un long run, noter quelles couches sont activées
   - Garder une copie du CSV de résultats avec un nom descriptif

2. **Analyser progressivement**:
   - Commencer par 1 backbone, quelques couches
   - Élargir progressivement aux autres backbones

3. **Comparer de manière équitable**:
   - Garder le même nombre d'images (n_inference, n_evaluation)
   - Utiliser le même metric (MMD avec RBF kernel)
   - Comparer dans les mêmes conditions

---

## 7. Structure du Projet

```
Métrique/
├── config.py              # Configuration principale + ENABLED_LAYERS
├── main.py                # Point d'entrée, orchestration
├── feature_extraction.py  # Extraction de features des backbones
├── distance_metrics.py    # Métriques MMD/FID
├── evaluation.py          # Évaluation de monotonie
├── degradation_generator.py  # Génération de dégradations
├── LAYERS_GUIDE.md        # Documentation des couches (ce fichier)
├── RUNNING_EXPERIMENTS.md # Guide d'utilisation (ce fichier)
├── results/               # Résultats CSV et graphiques
│   ├── cache/             # Cache d'activations (optionnel)
│   └── resultats_*.csv    # Fichiers de résultats
└── dataset/               # Images VisDrone2019
```

---

## 8. Questions Fréquentes

**Q: Combien de temps prend une expérimentation complète ?**
A: Avec 4 backbones × 6 layers × 100 ref images × 20 eval images = ~32 configs, comptez 6-12h selon le hardware (GPU recommandé pour DinoV2/CLIP).

**Q: Puis-je interrompre une expérimentation en cours ?**
A: Oui (Ctrl+C). Les résultats seront sauvegardés pour les configurations déjà complétées. Vous pouvez reprendre en désactivant les couches déjà testées.

**Q: Quelle est la différence entre mode medium et full ?**
A: Medium utilise seulement MMD/FID (distribution-free, plus rapide). Full inclut aussi des métriques basées sur GMM (Mahalanobis, neg_loglik) qui sont plus lentes.

**Q: Comment savoir quelles couches sont activées avant de lancer ?**
A: Utilisez:
```bash
python -c "from config import get_enabled_experiment_configs; configs = get_enabled_experiment_configs(); print(f'{len(configs)} configs:'); [print(f\"  {c['backbone']} layer {c['layer']}\") for c in configs]"
```

**Q: Puis-je utiliser mes propres images ?**
A: Oui, modifiez `CONFIG["dataset_path"]` dans config.py pour pointer vers votre dossier d'images. Format supporté: JPG, PNG.

**Q: Que signifie "gram_spatial" ?**
A: Calcul de la matrice de Gram (corrélations entre canaux) sur l'ensemble de la feature map spatiale. Capture les patterns de texture (style transfer, Gatys et al.).

**Q: Pourquoi MMD plutôt que FID ?**
A: MMD avec kernel RBF est plus robuste pour comparer des distributions avec peu d'échantillons (N_E=20). FID suppose des distributions gaussiennes.

---

## Ressources Additionnelles

- **LAYERS_GUIDE.md** : Documentation détaillée de toutes les couches extractables
- **config.py** : Fichier de configuration principal avec commentaires
- **README.md** : Vue d'ensemble du projet (si disponible)

---

**Dernière mise à jour:** 2026-02-12
**Auteur:** Documentation générée pour le projet Métrique
**Version:** 1.0
