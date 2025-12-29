# Mistral-MiniLLM_2025

**Mistral-MiniLLM_2025** est un projet d'apprentissage et découverte personnel visant à coder et entraîner un **petit modèle de langage français** inspiré des architectures modernes (GPT-like) mais entraînable localement.

Motivations :
- comprendre les mécanismes internes et d’un LLM ainsi que ce qui gravite autour (tokenizer, architecture, entraînement, génération) ;
- essayer à l'avenir d'en faire un mini agent conversationnel (si la phase de pré-training s'avère concluante).

---

## Le pojet actuellement

- **Corpus propre, diversifié et reproductible** (30M caractères).
- **Tokenizer BPE** adapté au français (littéraire, conversationnel, théâtre, dialogues restranscrits) avec gestion de tokens spéciaux pour les rôles user/agent.
- Entraînement du **LLM** (16M paramètres) sur (petit) GPU personnel.
- Pipeline complet : **prétraitement → tokenisation → entraînement → génération**.

---

## Fonctionnalités principales

### **Corpus** (`create_corpus.py`)
- Extraction de sources publiques (théâtre, dialogues, littérature courte).
- Nettoyage, normalisation, segmentation.
- Fichier Python de constitution de ce corpus automatiquement.

### **Tokenizer BPE** (`tokenizer.py`)
- Adapté d'un tutoriel d'Andrej Karpathy.
- Construction d’un vocabulaire optimisé pour le français.
- Gestion des caractères spéciaux, dialogues, ponctuation.
- Export du vocabulaire et des merges pour réutilisation sous format JSON.

### **Modèle** (`model.py`)
- Adapté d'un tutoriel d'Andrej Karpathy.
- Architecture d'un petit transformer basé sur la self-attention :
  - embeddings
  - multi-head attention
  - feed-forward
  - normalisation
- Taille adaptable (≈ 10–20M paramètres).

### **Entraînement** (`main_train.py`)
- Boucle d’entraînement simple et lisible.
- Gestion des checkpoints.
- Suivi de la loss avec TensorBoard.
- Génération périodique pour évaluer la progression.

### **Génération**
- Script provisoire pour tester le modèle entraîné.

---

## Résultats d’entraînement 

Cette section sera enrichie au fur et à mesure.

---

## Lancer l'entraînement

1. Générer le corpus (non intégré dans le script d'entraînement) : `python create_corpus.py`

2. Vérifier/ajuster les hyperparamètres dans : `config.py`

3. Lancer l'entraînement principal : `main_train.py`
   Le script charge le corpus, crée/charge le tokenizer, prépare le dataset, instancie le modèle, gère les checkpoints et démarre TensorBoard automatiquement.



