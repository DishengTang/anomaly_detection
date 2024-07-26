# MICA: Multi-channel Representation Refinement Contrastive Learning for Graph Fraud Detection

## Abstract:
Detecting fraudulent nodes from topological graphs is important in many real applications, such as financial fraud detection. This task is challenging due to both the class imbalance issue and the camouflaged behaviors of anomalous nodes. Recently, some graph contrastive learning (GCL) methods have been proposed to solve the above issue. However, local aggregation-based GNN encoders can not consider the long-distance nodes, leading to over-smoothing and false negative samples. Also, random perturbation data augmentation hinders separately considering camouflaged behaviors at the topological and feature levels. To address that, this paper proposes a novel contrastive learning architecture for enhancing the performance of graph fraud detection. Specifically, a context generator and a representation refinement module are embraced for mitigating the limitation of local aggregation in finding long-distance fraudsters, as well as the introduction of false negative samples in GCL. Further, a multi-channel fusion module is designed to adaptively defend against diverse camouflaged behaviors. The experimental results on real-world datasets show a significant performance improvement over baselines, which demonstrates its effectiveness.

## Paper Link

The full paper can be accessed [here](https://link.springer.com/chapter/10.1007/978-981-97-2421-5_3).

## Citation

If you use this code or data in your research, please cite our paper:
```bibtex
@inproceedings{wang2023mica,
  title={MICA: multi-channel representation refinement contrastive learning for graph fraud detection},
  author={Wang, Guifeng and Tang, Disheng and Shatsila, Anatoli and Zhang, Xuecang},
  booktitle={Asia-Pacific Web (APWeb) and Web-Age Information Management (WAIM) Joint International Conference on Web and Big Data},
  pages={31--46},
  year={2023},
  organization={Springer}
}
```
