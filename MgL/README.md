# Certainty from Uncertainty: Multi-granularity Labeling Inspired by Quantum Collapse for Learning with Noisy Labels in Fault Diagnosis 

![image](https://github.com/user-attachments/assets/7520be17-714f-4fc1-bd2e-343140fc2bcb)

MgL begins by mapping input signals to a latent feature space through the backbone of the diagnostic model. For each sample $x_i$, a corresponding feature cluster $\mathcal{C}_i$ is formed, and the feature distribution along with the label information within each cluster are combined to generate a superposition state $\varphi_i$, representing the cluster-center sample. Following the Copenhagen interpretation of quantum mechanics, the superposition state $\varphi$ collapses into a specific eigenstate $\phi_j$, resulting in a collapsed label. Meanwhile, a greedy algorithm is employed to select the eigenstate with the highest probability amplitude as the pseudo-label for the cluster-center sample. Finally, MgL re-labels the dataset by fusing the original annotated labels $\mathcal{Y}$, collapsed labels $\overline{\mathcal{Y}}$, and pseudo-labels $\mathcal{Y}'$, and these newly generated multi-granular labels $\widetilde{\mathcal{Y}}$ are then used as supervision to train the diagnostic model.


```
@ARTICLE{Dunkin3567264,
  author={Dunkin, Fir and Li, Xindeand Zhang, Zhentong and Wang, Kui and Gao, Tianrong and Wu, Guoliang and Li, Zhijun},
  journal={IEEE Transactions on Industrial Informatics}, 
  title={Certainty from Uncertainty: Multi-granularity Labeling Inspired by Quantum Collapse for Learning with Noisy Labels in Fault Diagnosis}, 
  year={2025},
  volume={Early Access},
  pages={1-12},
  keywords={Learning with noisy labels;Information fusion;Confirmation bias;Granular computing;Time series classification},
  doi={10.1109/TII.2025.3567264}
}
```
