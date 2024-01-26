# ImagiQ Federated

## Communication Prototype

All the communication between institutions (nodes) will be based on the designated folder and ssh connection. Each node will have `in` and `out` folders. In both of the folders, it will contain designated folders for each of the other institutions who wants to participate the federated learning. 

For example, Let's say there are 4 institutions (A, B, C, and D)  participating 4 institutions participating federated learning. In NodeA, it will have `in/B/`, `in/C/`, `in/D`, `out/B`, `out/C`, and `out/D`. Similary, in NodeD, it will have `in/A/`, `in/B/`, `in/C/`, `out/A/`, `out/B`, and `out/C`. Other nodes will have similar architecture of `in` and `out` folders.

### model sharing
If you want to share your models to all of the other institutions, you would place your model (`classifier_A`) in the `out/*` if you want to broadcast or `out/B/` if you want to share specifically to the NodeB. `classifier_A` will be placed at the other nodes `in/A` folder. Shared model will contain some `header` when sharing

### model receiving & model request
Others shared models will be held on the `in` folder. We can check if there's any incoming shared model from other institutions by checking if there is any content in each of the `in` folder. Other institutions might request your model by droping a signal at your `in` folder. 

![network protype](../../imgs/networkProtype.png)


<!-- References -->
[sea]:https://dollar.biz.uiowa.edu/~nstreet/research/kdd01.pdf
