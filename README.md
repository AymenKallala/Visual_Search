# Visual Search - 2023 - Object Detection Part

Repo contenant tout le travail d'exploration fait en Aout 2023. Le travail mené jusqu'à ce jour  concerne uniquement la partie 'Object Detection'. Aucun travail n'a encore été effectué concernant la partie 'Similarity Search'. 

Ce repo contient toutes les informations nécessaires pour entrainer des modèles YOLO (v5 ou v8) ou RT-DETR from scratch, en finetuning ou en transfer learning sur un dataset d'images custom dans un objectif de détection des objets.
 

# Installation de l'environnement
Pour setup l'environnement : Une fois le repo ayant été cloné :
1.  `python -m pip install poetry`
2.	`poetry install`
3.	`poetry update`

# Dataset

Le travail d'exploration a été effectué en expérimentant sur le dataset [FashionPedia](https://huggingface.co/datasets/detection-datasets/fashionpedia_4_categories/viewer/detection-datasets--fashionpedia_4_categories/train?row=0), un dataset opensource qui met à disposition environ 47000 images de produits "Fashion" en contexte. La version du dataset considérée ici contient 4 types de produits annotés : `Bags`, `Accessories`, `Clothing`, `Shoes`.
![alt-text-1](rm_images/train_examples.png "Example of annotated images in training set")

# Modèles

Ce repo se base sur l'API `utralytics` qui permet d'avoir accès à plusieurs architectures de [modèles](https://docs.ultralytics.com/models/) pré-entrainés sur des taches de détection d'objets.

## Yolos
D'un intérêt particulier a été la gamme [Yolov8](https://docs.ultralytics.com/models/yolov8/), mise à disposition sous plusieurs versions de taille différentes, pré-entrainés sur le dataset COCO.
![](rm_images/yolov8_archi.jpeg "Yolov8 available")


![alt-text-1](rm_images/yolo_sizes.png "Yolov8 available")

## RTDETR
Des essais ont également été réalisés sur des modèles Transformers [RTDETR](https://docs.ultralytics.com/models/rtdetr/#overview) également disponibles sur la plateforme ultralytics.

![alt-text-1](rm_images/rtdetr_archi.png "Yolov8 available")

# Configurer les données

Il est absolument MANDATOIRE que les données considérées soient au format YOLO. [Cliquez pour en savoir plus](https://docs.ultralytics.com/datasets/detect/)

Ce format passe par :
- un agencement du folder de données particuliers (un folder `images/` et `labels/` ainsi qu'un fichier `instances.json` par set de données).
- la rédaction d'un fichier `config.yaml` qui indique l'ensemble des labels ainsi que les différents PATHS vers les datasets de train, de validation et de test. [EXEMPLE](project_visual_search/data/configs/fashionpedia.yaml)

## Télécharger FashionPedia depuis HuggingFace

### Images et annotations

```
cd project_visual_search/data/fashionpedia
python fashionpedia_to_coco.py --data_path <DATA_FOLDER> --set train
python fashionpedia_to_coco.py --data_path <DATA_FOLDER> --set test
python fashionpedia_to_coco.py --data_path <DATA_FOLDER> --set val
```
Bien préciser à chaque fois: 
- `data_path` : l'endroit ou vous voulez stocker votre jeu de données, exemple : `/project-visual-search/datasets/fashionpedia`
- `set` : train, test ou val

### Générer les labels en fichier .txt par image

```
cd project_visual_search/data/fashionpedia
python create_labels.py --data_path <DATA_FOLDER_TRAIN>
python create_labels.py --data_path <DATA_FOLDER_TEST> 
python create_labels.py --data_path <DATA_FOLDER_VAL> 
```

Faire attention de bien générer les labels 3 fois (pour le folder de train, de test et de validation). Il est nécessaire que le `<DATA_FOLDER_PATH>` indiqué contienne un dossier `images/`.






# Entrainement sur dataset custom

## Finetuning

Afin de lancer le finetuning d'un modèle au choix : (Par exemple un yolov8 small)
```
python project_visual_search/yolo_v8/finetuning.py --model yolov8s.pt --data_path /data/configs/fashionpedia.yaml --batch_size 32  --epochs 100 --save_period 5 --lr 0.001 

```
- le `model` est à choisir parmis `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`. Les poids seront automatiquement téléchargés depuis l'API.
- le `data_path` doit pointer vers le `.yaml` décrivant le dataset au format YOLO (Nécessaire).

Pour lancer l'entrainement d'un modèle RTDETR :
```
python project_visual_search/detr/finetuning.py --model rtdetr-l.pt --batch_size 32  --epochs 100 --save_period 5 --lr 0.001 

```
le `model` est à choisir parmis `rtdetr-l.pt` et `rtdetr-x.pt`.
### Résultats 
A l'issue de l'entrainement un folder contenant tous les résultats, les poids et les logs sera crée dans le dossier `runs/`.

![](rm_images/confusion_matrix_normalized.png "Confusion Matrix")

![alt-text-3](rm_images/PR_curve.png "Courbe Precision-Recall")

## Transfer Learning (freeze quelques couches)

Le même procédé est à effectuer, seulement il faut préciser le nombre de "block" que l'on veut freeze dans notre réseau à l'aide du paramètre `freeze`.

```
python project_visual_search/detr/transfer_learning.py --model rtdetr-l.pt --freeze 20

```
**Pour information** le paramètre `freeze` n'est pas inclusif, pour un modèle de 23 modules comme tous les modèles yolo (numérotés sur PyTorch de 0 à 22), paramétrer `--freeze 22` gêlera toutes les layers de 0 à 21. Mais la layer numérotée 22 (la dernière) ne le sera pas.

**Concernant les modèles YOLOv8**. La **Detect Head** correspond au dernier module du modèle, donc le numéro 22.

Voici un aperçu de la taille de la detect head pour chaque taille de YOLO en nombre de paramètres.

| Yolo    | Total number of Params | Params in the Detect heads |
| -------- | ------- | --------|
| Nano| 3.1M|   800k  |
| Small| 11.1M| 2.14M|
| Medium| 25.9M|  3.8M |
| Large| 43.6M|  5.6M |
| XL| 59.4| 8.7M  |

# Evaluer un modèle

Pour procéder à la validation d'un modèle YOLO qu'on vient d'entrainer : (identique pour un RTDETR)

```
cd project_visual_search/yolo_v8
python validation.py --run <PATH TO THE MODEL>
```

`PATH TO THE MODEL` est par exemple `runs/train2`. Un folder contenant en son sein un dossier `weights` avec les poids acquis lors de l'entrainement. Il n'y a pas besoin de précisier le path du dataset de validation, il a déjà été enregistrer par l'API Ultralytics lors de l'entrainement du modèle (grâce au `config.yaml`).


![](rm_images/exemple_validation.png "VALIDATION")


# Générer des prédictions

```
cd project_visual_search/yolo_v8 (ou detr)
python predict.py --run <RUN NAME> --data_path <IMAGE_FOLDER_PATH> --conf 0.25 --output_dir <DIR_TO_STORE_RESULTS>
 
 ```
 - `run` Tout comme lors de l'évaluation, il faut spécifier le dossier de run correspondant au modèle que l'on veut utiliser pour générer des prédictions.
 - `data_path` Le folder d'images sur lequel on veut générer des prédictions de bounding boxes.
 - `conf` est le seuil de confiance à partir duquel le modèle génère une bounding box.
 - `output_dir` le folder (qui sera crée automatiquement) ou l'on veut stocker les prédictions annotées. exemple : `/home/aka/visual-search/project-visual-search/runs/predict/predictions_1`

## Exemple de prédictions générées :

 ![](rm_images/pred_generation_1.png "PREDICTION")
  ![](rm_images/pred_generation_2.png "PREDICTION")



 # Next steps

 - Set up MLFlow pour un meilleur tracking des expériences menées
 - Ajouter de la flexibilité dans la data augmentation utilisée lors du training (step essentiel pour une meilleur performance sur des données test + random)
 
 