# Third-party data licenses and attribution

Processed inputs preserve the original aligned node ordering and evaluation masks. Changes comprise conversion to the stored sparse graph/evaluation format and the recorded layer preprocessing. The model's MIT license does not relicense these datasets.

| Processed dataset | Original source | Terms |
|---|---|---|
| AUCS | https://github.com/CoMuNeLab/CS-Aarhus | ODbL 1.0 and Database Contents License 1.0; notices in its data folder and LICENSES/ |
| Lazega | https://github.com/CoMuNeLab/Lazega-Law-Firm | ODbL 1.0 and Database Contents License 1.0 |
| CKM | https://github.com/CoMuNeLab/CKM-Physicians-Innovation | ODbL 1.0 and Database Contents License 1.0 |
| arXiv NetScience | https://github.com/CoMuNeLab/arXiv-Network-Science-Multiplex | ODbL 1.0 and Database Contents License 1.0 |
| HighSchool | https://sociopatterns.org/datasets/high-school-contact-and-friendship-networks/ | CC BY-NC-SA 3.0, https://creativecommons.org/licenses/by-nc-sa/3.0/ |
| PrimarySchool | https://sociopatterns.org/datasets/primary-school-temporal-network-data/ | CC BY-NC-SA 3.0, https://creativecommons.org/licenses/by-nc-sa/3.0/ |
| Workplace | https://sociopatterns.org/datasets/test/ | CC0 1.0, https://creativecommons.org/publicdomain/zero/1.0/ |
| UCI Multiple Features | Robert Duin, https://archive.ics.uci.edu/dataset/72/multiple+features, DOI https://doi.org/10.24432/C5HC70 | CC BY 4.0, https://creativecommons.org/licenses/by/4.0/ |

We acknowledge the SocioPatterns collaboration (https://www.sociopatterns.org). When using HighSchool, cite Mastrandrea et al., PLoS ONE 10(9), e0136497 (2015), https://doi.org/10.1371/journal.pone.0136497. For PrimarySchool, cite Gemmetto et al., BMC Infectious Diseases 14:695 (2014), and Stehle et al., PLoS ONE 6(8), e23176 (2011). For Workplace, cite Genois and Barrat, EPJ Data Science 7, 11 (2018).

These terms also apply to copies of the same processed inputs in experiments/*/data and the shared-template runtime. In particular, HighSchool and PrimarySchool data remain noncommercial/share-alike; original CT-GWAE software remains MIT.

The processed DK and MNIST-derived arrays are acquired separately; no new redistribution permission is asserted here. Their frozen identities and source information are recorded in configs/catalog.json and docs/datasets.md. Our synthetic graph inputs and original numerical reference outputs are original project materials covered by the repository's MIT license.
