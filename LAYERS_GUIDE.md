# Guide d'Extraction des Couches (Layer Extraction Guide)

## Vue d'ensemble

Ce guide documente toutes les couches disponibles pour l'extraction de features dans chaque backbone model. Il explique quelles couches peuvent être utilisées pour extraire des features d'activation et pourquoi certaines couches sont omises.

### Qu'est-ce qu'une couche extractable ?

Une **couche extractable** est une couche du réseau de neurones qui produit des représentations de features significatives. Ces couches incluent typiquement :
- **Conv2d** : Couches convolutionnelles qui capturent des patterns visuels
- **Linear** : Couches fully-connected qui capturent des représentations sémantiques
- **Transformer blocks** : Blocs d'attention qui capturent des relations contextuelles
- **ResNet blocks** : Blocs résiduels qui capturent des features hiérarchiques
- **LayerNorm/BatchNorm** : Couches de normalisation qui stabilisent les features

### Pourquoi certaines couches sont omises ?

Certaines couches sont **non-extractables** car elles ne fournissent pas d'informations de texture/distribution indépendantes :
- **ReLU/GELU** : Activations qui ne modifient que les valeurs (in-place)
- **MaxPool/AvgPool** : Downsampling spatial sans nouvelle information
- **Dropout** : Couche de régularisation aléatoire
- **Reshape/Flatten** : Operations géométriques sans transformation de features

### Comment utiliser ce guide avec config.py

1. Consultez les tables ci-dessous pour identifier les couches qui vous intéressent
2. Notez l'index de la couche
3. Dans `config.py`, trouvez `ENABLED_LAYERS` (ligne ~312)
4. Changez `True`/`False` pour activer/désactiver la couche

---

## 1. SD VAE (Stable Diffusion VAE) - 17 couches extractables

**Model size:** ~167M parameters
**Input:** 256×256×3 RGB image
**Output:** 32×32×8 latent space
**Architecture:** Hierarchical autoencoder with residual blocks and attention

| Index | Layer Name | Type | Channels | Resolution | Semantic Level | Description |
|-------|------------|------|----------|------------|----------------|-------------|
| 0 | `encoder.conv_in` | Conv2d | 128 | 256×256 | **Early** | Initial convolution - captures low-level textures |
| 1 | `encoder.down_blocks.0.resnets.0` | ResNet | 128 | 256×256 | Early | First residual block - refines early features |
| 2 | `encoder.down_blocks.0.resnets.1` | ResNet | 128 | 256×256 | Early | Second residual block - further refinement |
| 3 | `encoder.down_blocks.0.downsamplers.0` | Conv2d | 128 | 128×128 | **Early-Mid** | Downsamples to 128×128 - reduces spatial resolution |
| 4 | `encoder.down_blocks.1.resnets.0` | ResNet | 256 | 128×128 | Early-Mid | Doubles channels to 256 - increased capacity |
| 5 | `encoder.down_blocks.1.resnets.1` | ResNet | 256 | 128×128 | Early-Mid | Refines 256-channel features |
| 6 | `encoder.down_blocks.1.downsamplers.0` | Conv2d | 256 | 64×64 | **Mid** | Downsamples to 64×64 - medium semantic level |
| 7 | `encoder.down_blocks.2.resnets.0` | ResNet | 512 | 64×64 | Mid | Doubles channels to 512 - high capacity |
| 8 | `encoder.down_blocks.2.resnets.1` | ResNet | 512 | 64×64 | Mid | Refines 512-channel features |
| 9 | `encoder.down_blocks.2.downsamplers.0` | Conv2d | 512 | 32×32 | **Mid-Late** | Downsamples to 32×32 - higher semantics |
| 10 | `encoder.down_blocks.3.resnets.0` | ResNet | 512 | 32×32 | Mid-Late | Maintains 512 channels at 32×32 |
| 11 | `encoder.down_blocks.3.resnets.1` | ResNet | 512 | 32×32 | Mid-Late | Final downsample block refinement |
| 12 | `encoder.mid_block.resnets.0` | ResNet | 512 | 32×32 | **Late** | Bottleneck: first mid-block residual |
| 13 | `encoder.mid_block.attentions.0` | Attention | 512 | 32×32 | Late | Self-attention - global context |
| 14 | `encoder.mid_block.resnets.1` | ResNet | 512 | 32×32 | Late | Bottleneck: second mid-block residual |
| 15 | `encoder.conv_norm_out` | GroupNorm | 512 | 32×32 | Late | Final normalization before latent |
| 16 | `encoder.conv_out` | Conv2d | 8 | 32×32 | **Final** | Latent space representation |

**Non-extractable layers:**
- Internal ResNet components (norm1, conv1, norm2, conv2) - redundant to block outputs
- Decoder path - not used (only encoder is relevant)
- down_blocks.3 has NO downsampler (stays at 32×32)

**Recommended layers for experiments:** 0, 3, 6, 9, 12, 16 (early, early-mid, mid, mid-late, late, final)

---

## 2. DinoV2 ViT-B/14 - 14 couches extractables

**Model size:** ~86M parameters
**Input:** 518×518×3 RGB image
**Output:** 1369 tokens × 768 dimensions
**Architecture:** Vision Transformer with 12 transformer blocks

| Index | Layer Name | Type | Features | Tokens | Semantic Level | Description |
|-------|------------|------|----------|--------|----------------|-------------|
| 0 | `patch_embed` | PatchEmbed | 768 | 1369 | **Initial** | Converts 518×518 image to 37×37 patches (14×14 each) |
| 1 | `blocks.0` | Transformer | 768 | 1369 | Early | First transformer block - local patterns |
| 2 | `blocks.1` | Transformer | 768 | 1369 | **Early-Mid** | Second block - emerging patterns |
| 3 | `blocks.2` | Transformer | 768 | 1369 | Early-Mid | Third block - pattern refinement |
| 4 | `blocks.3` | Transformer | 768 | 1369 | Early-Mid | Fourth block - increased abstraction |
| 5 | `blocks.4` | Transformer | 768 | 1369 | **Mid** | Fifth block - mid-level semantics |
| 6 | `blocks.5` | Transformer | 768 | 1369 | Mid | Sixth block - semantic consolidation |
| 7 | `blocks.6` | Transformer | 768 | 1369 | Mid | Seventh block - deeper semantics |
| 8 | `blocks.7` | Transformer | 768 | 1369 | **Mid-Late** | Eighth block - higher abstraction |
| 9 | `blocks.8` | Transformer | 768 | 1369 | Mid-Late | Ninth block - complex patterns |
| 10 | `blocks.9` | Transformer | 768 | 1369 | Mid-Late | Tenth block - refined semantics |
| 11 | `blocks.10` | Transformer | 768 | 1369 | **Late** | Eleventh block - high-level features |
| 12 | `blocks.11` | Transformer | 768 | 1369 | Late | Twelfth block - final transformer |
| 13 | `norm` | LayerNorm | 768 | 1369 | **Final** | Final normalization layer |

**Non-extractable layers:**
- `head` (Identity) - classification head not used
- Internal attention heads - captured by block outputs
- Intermediate FFN layers - captured by block outputs

**Notes:**
- All transformer blocks are included because DinoV2 uses self-supervised learning where each block contributes
- Output is (B, 1369, 768) - averaged over tokens to get (B, 768) for Gram computation

**Recommended layers for experiments:** 0, 2, 5, 8, 11, 13 (initial, early-mid, mid, mid-late, late, final)

---

## 3. VGG19 - 18 couches extractables

**Model size:** ~144M parameters
**Input:** 224×224×3 RGB image
**Output:** 1000-class logits
**Architecture:** Sequential convolutional network with 5 conv stages + 3 FC layers

| Index | Layer Name | Type | Channels | Resolution | Semantic Level | Description |
|-------|------------|------|----------|------------|----------------|-------------|
| 0 | `features.0` | Conv2d | 64 | 224×224 | **Very Early** | conv1_1 - first convolution, raw textures |
| 1 | `features.2` | Conv2d | 64 | 224×224 | Very Early | conv1_2 - refines raw textures |
| 2 | `features.5` | Conv2d | 128 | 112×112 | Early | conv2_1 - after maxpool, 128 channels |
| 3 | `features.7` | Conv2d | 128 | 112×112 | **Early-Mid** | conv2_2 - refines 128-ch features |
| 4 | `features.10` | Conv2d | 256 | 56×56 | Early-Mid | conv3_1 - doubles to 256 channels |
| 5 | `features.12` | Conv2d | 256 | 56×56 | Early-Mid | conv3_2 - refines 256-ch features |
| 6 | `features.14` | Conv2d | 256 | 56×56 | Early-Mid | conv3_3 - deeper 256-ch refinement |
| 7 | `features.16` | Conv2d | 256 | 56×56 | **Mid** | conv3_4 - final 256-ch layer in stage 3 |
| 8 | `features.19` | Conv2d | 512 | 28×28 | Mid | conv4_1 - doubles to 512 channels |
| 9 | `features.21` | Conv2d | 512 | 28×28 | Mid | conv4_2 - refines 512-ch features |
| 10 | `features.23` | Conv2d | 512 | 28×28 | Mid | conv4_3 - deeper 512-ch refinement |
| 11 | `features.25` | Conv2d | 512 | 28×28 | **Mid-Late** | conv4_4 - final layer in stage 4 |
| 12 | `features.28` | Conv2d | 512 | 14×14 | Mid-Late | conv5_1 - maintains 512 channels |
| 13 | `features.30` | Conv2d | 512 | 14×14 | Mid-Late | conv5_2 - refines at 14×14 |
| 14 | `features.32` | Conv2d | 512 | 14×14 | Late | conv5_3 - deeper refinement |
| 15 | `features.34` | Conv2d | 512 | 14×14 | **Late** | conv5_4 - final conv layer |
| 16 | `classifier.0` | Linear | 4096 | - | Late | FC1 - first fully-connected (25088→4096) |
| 17 | `classifier.3` | Linear | 4096 | - | **Final** | FC2 - second fully-connected (4096→4096) |

**Non-extractable layers:**
- ReLU activations (features.[1,3,6,8,11,13,15,17,20,22,24,26,29,31,33,35]) - redundant
- MaxPool2d layers (features.[4,9,18,27,36]) - spatial downsampling only
- Dropout layers (classifier.[2,5]) - regularization only
- classifier.6 (Linear 4096→1000) - task-specific classification head

**Notes:**
- VGG19 has 16 conv layers + 3 FC layers = 19 total
- Channel progression: 64 → 128 → 256 → 512 → 512
- Spatial resolution: 224 → 112 → 56 → 28 → 14 (after each maxpool)

**Recommended layers for experiments:** 0, 3, 7, 11, 15, 17 (very early, early-mid, mid, mid-late, late, final)

---

## 4. LPIPS VGG - 16 couches extractables

**Model size:** ~14M parameters (VGG backbone only, no LPIPS linear layers)
**Input:** 224×224×3 RGB image (normalized to [-1, 1])
**Output:** 16 conv layer outputs for LPIPS distance
**Architecture:** VGG-like backbone divided into 5 slices

| Index | Layer Name | Type | Channels | Resolution | Semantic Level | Description |
|-------|------------|------|----------|------------|----------------|-------------|
| 0 | `net.slice1.0` | Conv2d | 64 | 224×224 | **Very Early** | conv1_1 - first convolution |
| 1 | `net.slice1.2` | Conv2d | 64 | 224×224 | Very Early | conv1_2 - refines raw textures |
| 2 | `net.slice2.5` | Conv2d | 128 | 112×112 | Early | conv2_1 - after maxpool |
| 3 | `net.slice2.7` | Conv2d | 128 | 112×112 | **Early-Mid** | conv2_2 - refines 128-ch |
| 4 | `net.slice3.10` | Conv2d | 256 | 56×56 | Early-Mid | conv3_1 - doubles to 256 |
| 5 | `net.slice3.12` | Conv2d | 256 | 56×56 | Early-Mid | conv3_2 - refines 256-ch |
| 6 | `net.slice3.14` | Conv2d | 256 | 56×56 | Early-Mid | conv3_3 - deeper refinement |
| 7 | `net.slice3.16` | Conv2d | 256 | 56×56 | **Mid** | conv3_4 - final in stage 3 |
| 8 | `net.slice4.19` | Conv2d | 512 | 28×28 | Mid | conv4_1 - doubles to 512 |
| 9 | `net.slice4.21` | Conv2d | 512 | 28×28 | Mid | conv4_2 - refines 512-ch |
| 10 | `net.slice4.23` | Conv2d | 512 | 28×28 | Mid | conv4_3 - deeper refinement |
| 11 | `net.slice4.25` | Conv2d | 512 | 28×28 | **Mid-Late** | conv4_4 - final in stage 4 |
| 12 | `net.slice5.28` | Conv2d | 512 | 14×14 | Mid-Late | conv5_1 - maintains 512 |
| 13 | `net.slice5.30` | Conv2d | 512 | 14×14 | **Late** | conv5_2 - refines at 14×14 |
| 14 | `net.slice5.32` | Conv2d | 512 | 14×14 | Late | conv5_3 - deeper refinement |
| 15 | `net.slice5.34` | Conv2d | 512 | 14×14 | **Final** | conv5_4 - final conv layer |

**Non-extractable layers:**
- ReLU activations within slices - redundant
- MaxPool2d layers (beginning of slices 2-5) - spatial downsampling only
- `scaling_layer` - normalization preprocessing
- `lin0-lin4` (Linear projection layers) - LPIPS-specific metric computation

**Notes:**
- LPIPS VGG uses the SAME architecture as VGG19 conv layers
- Layer indices in slice names reflect original VGG sequential positions (e.g., slice1.0, slice1.2 not slice1.0, slice1.1)
- No FC layers - only conv layers used for perceptual distance
- Channel progression identical to VGG19: 64 → 128 → 256 → 512 → 512

**Recommended layers for experiments:** 0, 3, 7, 11, 13, 15 (very early, early-mid, mid, mid-late, late, final)

---

## 5. ResNet50 - 10 couches extractables

**Model size:** ~25M parameters
**Input:** 224×224×3 RGB image
**Output:** 1000-class logits
**Architecture:** Residual network with 4 stages of bottleneck blocks

| Index | Layer Name | Type | Channels | Resolution | Semantic Level | Description |
|-------|------------|------|----------|------------|----------------|-------------|
| 7 | `layer3` | Bottleneck×6 | 1024 | 14×14 | **Mid** | Stage 3 - 6 bottleneck blocks (512→1024) |
| 8 | `layer4` | Bottleneck×3 | 2048 | 7×7 | **Late** | Stage 4 - 3 bottleneck blocks (1024→2048) |
| 9 | `avgpool` | AdaptiveAvgPool2d | 2048 | 1×1 | Late | Global average pooling to (B, 2048) |
| 10 | `fc` | Linear | 1000 | - | Final | Classification head (2048→1000) |
| 11 | `layer4.0` | Bottleneck | 2048 | 7×7 | Late | First bottleneck in stage 4 |
| 12 | `layer4.1` | Bottleneck | 2048 | 7×7 | Late | Second bottleneck in stage 4 |
| 13 | `layer4.2` | Bottleneck | 2048 | 7×7 | Late | Third bottleneck in stage 4 |
| 14 | `layer3.5` | Bottleneck | 1024 | 14×14 | Mid | Sixth bottleneck in stage 3 |
| 15 | `layer3.4` | Bottleneck | 1024 | 14×14 | Mid | Fifth bottleneck in stage 3 |

**Non-extractable layers (not in config but exist in model):**
- `conv1` (Conv2d 3→64, 7×7, stride=2) - initial convolution
- `bn1` (BatchNorm2d) - batch normalization
- `relu` (ReLU) - activation
- `maxpool` (MaxPool2d 3×3, stride=2) - spatial downsampling
- `layer1` (Bottleneck×3, 256 channels) - stage 1, too early
- `layer2` (Bottleneck×4, 512 channels) - stage 2, redundant with layer3
- Individual conv layers within bottleneck blocks (conv1, conv2, conv3) - redundant to block outputs

**Notes:**
- ResNet50 has 4 main stages: layer1 (256ch), layer2 (512ch), layer3 (1024ch), layer4 (2048ch)
- Only layer3 and layer4 are included - earlier layers too low-level
- Bottleneck block structure: 1×1 conv → 3×3 conv → 1×1 conv with skip connection
- Individual bottleneck indices (11-15) provide finer granularity within layer3/layer4

**Recommended layers for experiments:** 7, 8, 9 (mid, late, global pooling)

---

## 6. CLIP ViT-Base - 13 couches extractables

**Model size:** ~87M parameters
**Input:** 224×224×3 RGB image
**Output:** 50 tokens × 768 dimensions (49 patches + 1 CLS token)
**Architecture:** Vision Transformer with 12 encoder layers

| Index | Layer Name | Type | Features | Tokens | Semantic Level | Description |
|-------|------------|------|----------|--------|----------------|-------------|
| 0 | `vision_model.embeddings` | PatchEmbed | 768 | 50 | **Initial** | Patch embedding (32×32 patches) + positional encoding |
| 1 | `vision_model.encoder.layers.0` | Transformer | 768 | 50 | Early | First transformer block - local patterns |
| 2 | `vision_model.encoder.layers.1` | Transformer | 768 | 50 | Early | Second block - emerging patterns |
| 3 | `vision_model.encoder.layers.2` | Transformer | 768 | 50 | **Early-Mid** | Third block - pattern refinement |
| 4 | `vision_model.encoder.layers.3` | Transformer | 768 | 50 | Early-Mid | Fourth block - increased abstraction |
| 5 | `vision_model.encoder.layers.4` | Transformer | 768 | 50 | Early-Mid | Fifth block - mid-level semantics |
| 6 | `vision_model.encoder.layers.5` | Transformer | 768 | 50 | **Mid** | Sixth block - semantic consolidation |
| 7 | `vision_model.encoder.layers.6` | Transformer | 768 | 50 | Mid | Seventh block - deeper semantics |
| 8 | `vision_model.encoder.layers.7` | Transformer | 768 | 50 | Mid | Eighth block - higher abstraction |
| 9 | `vision_model.encoder.layers.8` | Transformer | 768 | 50 | **Mid-Late** | Ninth block - complex patterns |
| 10 | `vision_model.encoder.layers.9` | Transformer | 768 | 50 | Mid-Late | Tenth block - refined semantics |
| 11 | `vision_model.encoder.layers.10` | Transformer | 768 | 50 | Mid-Late | Eleventh block - high-level features |
| 12 | `vision_model.encoder.layers.11` | Transformer | 768 | 50 | **Final** | Twelfth block - final transformer |

**Non-extractable layers:**
- `pre_layernorm` (LayerNorm) - input normalization
- `post_layernorm` (LayerNorm) - not used in current config (redundant normalization)
- Internal attention heads (12 per layer) - captured by layer outputs
- FFN intermediate states (3072-dim) - captured by layer outputs
- Individual CLS token vs patch tokens - averaged together for Gram computation

**Notes:**
- CLIP uses patch-32 tokenization (coarser than DinoV2's patch-14)
- 50 tokens = 49 image patches (7×7 grid) + 1 CLS token
- All transformer blocks contribute to vision-language alignment
- Output is (B, 50, 768) - averaged over tokens to get (B, 768)

**Recommended layers for experiments:** 0, 3, 6, 9, 12 (initial, early-mid, mid, mid-late, final)

---

## Résumé des Couches Extractables par Backbone

| Backbone | Total Layers | Extractable | Currently Enabled | Missing Potential | Reason for Omission |
|----------|--------------|-------------|-------------------|-------------------|---------------------|
| **SD VAE** | 17 | 17 | 6 (0,3,6,9,12,16) | 11 intermediate ResNets | Coarser hierarchical sampling sufficient |
| **DinoV2** | 14 | 14 | 6 (0,2,5,8,11,13) | 8 intermediate blocks | Self-supervised design benefits from sampling |
| **VGG19** | 18 | 18 | 6 (0,3,7,11,15,17) | 12 intermediate conv | Progressive sampling covers hierarchy |
| **LPIPS VGG** | 16 | 16 | 6 (0,3,7,11,13,15) | 10 intermediate conv | Similar to VGG19, progressive sampling |
| **ResNet50** | 10 | 10 | 3 (7,8,9) | layer1, layer2, sub-blocks | Early layers too low-level, sub-blocks redundant |
| **CLIP ViT** | 13 | 13 | 5 (0,3,6,9,12) | 8 intermediate blocks | Vision-language alignment benefits from sampling |

---

## Conseils pour la Sélection de Couches

### Stratégies de Sélection

1. **Échantillonnage hiérarchique (Recommended)**
   - Activer des couches à différents niveaux sémantiques (early, mid, late)
   - Exemple: SD VAE layers 0, 3, 6, 9, 12, 16
   - **Avantage:** Capture la progression de l'information à travers le réseau

2. **Comparaison précoce vs tardive**
   - Activer seulement early layers (0-3) OU late layers (12-16)
   - **Avantage:** Compare sensibilité aux textures vs sémantique

3. **Focus sur les transitions**
   - Activer les downsamplers (SD VAE: 3, 6, 9) et mid-block (12)
   - **Avantage:** Capture les points de compression d'information

4. **Expérimentation complète**
   - Activer TOUTES les couches (computationally expensive)
   - **Avantage:** Exploration exhaustive, mais long runtime

### Recommandations par Backbone

- **SD VAE:** Layers 0, 3, 6, 9, 12, 16 (6 layers, early → late)
- **DinoV2:** Layers 0, 2, 5, 8, 11, 13 (6 layers, initial → final)
- **VGG19:** Layers 0, 3, 7, 11, 15, 17 (6 layers, very early → final)
- **LPIPS VGG:** Layers 0, 3, 7, 11, 13, 15 (6 layers, very early → final)
- **ResNet50:** Layers 7, 8, 9 (3 layers, mid → pooled)
- **CLIP ViT:** Layers 0, 3, 6, 9, 12 (5 layers, initial → final)

**Total:** 32 configurations (24 for full sampling, 32 if all backbones enabled)

---

## Questions Fréquentes

**Q: Pourquoi certains indices sont manquants dans ResNet50 (0-6) ?**
A: ResNet50 config inclut seulement layer3-4 et sub-blocks. Les couches 0-6 (conv1, bn1, relu, maxpool, layer1, layer2) ne sont pas définies car trop bas-niveau ou redondantes.

**Q: Pourquoi LPIPS VGG a des indices étranges (slice1.0, slice2.5) ?**
A: Les indices reflètent les positions ORIGINALES dans VGG Sequential. Par exemple, slice1.0 = features.0 (conv1_1), slice1.2 = features.2 (conv1_2, car features.1 est ReLU).

**Q: Dois-je activer TOUTES les couches ?**
A: Non, c'est computationally expensive. Recommandé : 5-6 couches par backbone (early, mid, late) suffit pour capturer la hiérarchie.

**Q: Quelle est la différence entre VGG19 et LPIPS VGG ?**
A: Même architecture convolutionnelle, mais LPIPS VGG omet les FC layers et utilise une normalisation différente ([-1,1] au lieu de ImageNet stats).

**Q: Pourquoi DinoV2 et CLIP ont-ils 768 dimensions ?**
A: Taille standard des embeddings pour ViT-Base. ViT-Large utilise 1024, ViT-Huge utilise 1280.

---

**Dernière mise à jour:** 2026-02-12
**Auteur:** Documentation générée pour le projet Métrique
