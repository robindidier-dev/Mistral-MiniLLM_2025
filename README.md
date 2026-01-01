# Mistral-MiniLLM_2025

**Mistral-MiniLLM_2025** est un projet d'apprentissage et découverte personnel visant à coder et entraîner un **petit modèle de langage français** inspiré des architectures modernes (GPT-like).

Motivations :
- comprendre les mécanismes internes et d’un LLM ainsi que ce qui gravite autour (tokenizer, architecture, entraînement, génération) ;
- essayer à l'avenir d'en faire un mini agent conversationnel.

---

## Le pojet actuellement

- **Corpus propre, diversifié et reproductible** réparti en 5000 fichiers (140 Mo de texte de Wikipédia, classiques du théâtre et de la littérature).
- **Tokenizer BPE** adapté au français avec gestion de tokens spéciaux pour les rôles user/agent.
- Entraînement du **LLM** (16M paramètres) sur laptop personnel (upscale envisageable mais sur machine dédiée...).
- Pipeline complet : **prétraitement → tokenisation → entraînement → génération**.

---

## Fonctionnalités principales

### **Corpus** (`create_corpus.py`)
(Nouvelle version -> corpus 10x plus massif et encore plus normalisé)
- Extraction de sources publiques (théâtre, dialogues, littérature courte).
- Nettoyage, normalisation, segmentation (codé à l'aide de l'IA car l'étape de normalisation est en réalité très complexe).


### **Tokenizer BPE** (`tokenizer.py`)
(Nouvelle version -> fix des bugs de merges inutiles.)
- Adapté d'un tutoriel d'Andrej Karpathy.
- Construction d’un vocabulaire optimisé pour le français.
- Gestion des caractères spéciaux + tokens spéciaux qui pourront servir pour le fine-tuning et le mode 'agent conversationnel'.
- Export du vocabulaire et des merges pour réutilisation sous format JSON.

### **Modèle** (`model.py`)
(Nouvelle version -> Ajout d'un paramètre de température pour la génération)
- Adapté d'un tutoriel d'Andrej Karpathy.
- Architecture d'un petit transformer basé sur la self-attention :
  - embeddings
  - multi-head attention
  - feed-forward
  - normalisation
- Taille adaptable (≈ 10–20M paramètres).

### **Entraînement** (`main_train.py`)
(Nouvelle version -> Meilleure gestion des sauvegardes, micro-baches pour augmenter la taille du contexte)
- Gestion des checkpoints.
- Suivi des losses, learning rate & norme des gradients avec TensorBoard.

### **Génération**
(Nouvelle version -> Animation de génération séquentielle)
- Script provisoire pour tester le modèle entraîné.

---

## Résultats d’entraînement 

<p align="center">
  <img src="assets/Training.png" alt="Training loss & LR" width="48%">
  <img src="assets/Training_2.png" alt="LR & gradient norm" width="48%">
</p>




