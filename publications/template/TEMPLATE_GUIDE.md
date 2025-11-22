# Guide d'utilisation du Template LaTeX/Quarto

## Structure du Template

Ce template offre une mise en page professionnelle avec :
- Marges basées sur le nombre d'or (1.618 cm)
- En-tête avec titre à gauche et numéro de page à droite
- Première page avec titre/règle/auteur/abstract/règle
- Espacement optimal (1.2)
- Police lmodern

## Option 1 : Utilisation directe avec LaTeX

### Fichier : `template_overleaf.tex`

1. Ouvrir dans Overleaf ou votre éditeur LaTeX
2. Modifier les sections marquées "Modify this" :
   - Ligne 33 : `\fancyhead[L]{Your Document Title}`
   - Lignes 43-44 : titre et auteur dans hyperref
   - Lignes 70-72 : titre principal
   - Ligne 87 : informations auteur
   - Ligne 91+ : abstract

3. Compiler avec pdfLaTeX

### Personnalisation rapide

```latex
% Changer le titre dans l'en-tête
\fancyhead[L]{Mon Nouveau Titre}

% Ajouter des commandes mathématiques personnalisées
\newcommand{\R}{\mathbb{R}}
\newcommand{\N}{\mathbb{N}}

% Modifier l'espacement
\setstretch{1.5}  % Plus d'espace entre les lignes
```

## Option 2 : Utilisation avec Quarto/Markdown

### Fichier : `template_quarto_header.yml`

1. Créer un fichier `.qmd` ou `.md`
2. Copier le contenu YAML au début du fichier
3. Écrire le contenu en Markdown

### Exemple d'utilisation

```yaml
---
title: "Analyse Statistique"
subtitle: "Résultats préliminaires"
author: 
  - name: "Jean Dupont"
    affiliation: "Université XYZ"
    email: "jean.dupont@xyz.fr"
abstract: |
  Ceci est mon abstract...
  
  **Keywords**: statistiques, analyse, données
---

## Introduction

Mon contenu en Markdown...

### Sous-section

- Point 1
- Point 2

Équation : $E = mc^2$
```

### Compiler avec Quarto

```bash
quarto render document.qmd
```

### Compiler avec Pandoc

```bash
pandoc document.md -o output.pdf \
  --template=template_overleaf.tex \
  --pdf-engine=pdflatex
```

## Éléments clés du template

### En-tête et pied de page

```latex
\fancyhead[L]{Titre à gauche}
\fancyhead[R]{\thepage}  % Numéro de page à droite
\renewcommand{\headrulewidth}{0.2pt}  % Épaisseur du trait
```

### Page de titre avec règles

```latex
\maketitle
\noindent\rule{\textwidth}{0.2pt}  % Règle horizontale

% Informations auteur
{Nom\\
Affiliation\\
email}

\vfill  % Espace vertical flexible

\begin{abstract}
...
\end{abstract}

\vfill

\noindent\rule{\textwidth}{0.2pt}  % Règle de fin
```

### Tables professionnelles

```latex
\begin{table}[H]
\centering
\begin{tabular}{lll}
\toprule
\textbf{Col 1} & \textbf{Col 2} & \textbf{Col 3} \\
\midrule
Data 1 & Data 2 & Data 3 \\
\bottomrule
\end{tabular}
\caption{Description}
\end{table}
```

## Packages inclus

- **Mise en page** : geometry, fancyhdr, setspace
- **Mathématiques** : amsmath, amssymb
- **Tables** : booktabs, longtable, array
- **Figures** : float, caption, subcaption, graphicx
- **Liens** : hyperref (avec couleurs)
- **Graphiques** : tikz
- **Citations** : csquotes

## Personnalisation des marges

```latex
\usepackage[
  margin=1.618cm,    % Marge générale
  top=2.618cm,       # Marge haute
  bottom=2.618cm     # Marge basse
]{geometry}
```

## Conseils

1. **Pour Overleaf** : Télécharger `template_overleaf.tex` directement
2. **Pour Markdown** : Utiliser le YAML header avec Quarto
3. **Personnalisation** : Modifier les valeurs dans les sections marquées
4. **TOC** : Supprimer `\tableofcontents` si non désiré
5. **No emojis** : Le template respecte votre préférence (ton sobre)

## Compatibilité

- ✅ Overleaf
- ✅ TeXLive
- ✅ MiKTeX
- ✅ Quarto
- ✅ Pandoc 2.0+

## Structure minimale pour Markdown

Si vous voulez juste convertir du Markdown simple :

```markdown
---
title: "Mon Titre"
author: "Mon Nom"
date: "2025-01-01"
---

# Section 1

Contenu...
```

Puis :

```bash
pandoc input.md -o output.pdf \
  --pdf-engine=pdflatex \
  -V geometry:margin=1.618cm \
  -V geometry:top=2.618cm \
  -V geometry:bottom=2.618cm \
  -V fontfamily=lmodern \
  -V linestretch=1.2
```

