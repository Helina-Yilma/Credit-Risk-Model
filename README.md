# KAIM Week 4 Credit Risk Probability Model for Alternative Data
An End-to-End Implementation for Building, Deploying, and Automating a Credit Risk Model

### Credit Scoring Business Understanding

#### Influence of Basel II on Model Interpretability and Documentation

The Basel II Accord places strong emphasis on accurate risk measurement, transparency, and supervisory review, particularly through its Internal Ratings-Based (IRB) approaches. This regulatory focus increases the need for interpretable and well-documented credit risk models, as banks must be able to clearly explain how risk estimates are produced, justify model assumptions, and demonstrate reliability to regulators and internal stakeholders. An interpretable model supports validation, auditability, and governance, ensuring that risk decisions are consistent, defensible, and compliant with regulatory expectations.

#### Need for a Proxy Default Variable and Associated Risks

In our dataset as it is the case for many real-world datasets, a direct default label is unavailable due to data limitations or reporting constraints. As a result, it becomes necessary to create a proxy variable (e.g., 90+ days past due) to approximate default behavior. While this enables model development, it introduces model risk, as the proxy may not fully capture true credit default events. Predictions based on an imperfect proxy can lead to misclassification of borrowers, incorrect capital allocation, biased risk estimates, and suboptimal lending decisions. From a business perspective, this may increase credit losses, reduce profitability, or expose the institution to regulatory scrutiny if the proxy does not align with regulatory definitions of default.

#### Trade-offs Between Interpretable and High-Performance Models

There is a fundamental trade-off between model interpretability and predictive performance in regulated financial environments. Simple models such as Logistic Regression with Weight of Evidence (WoE) offer transparency, stability, and ease of explanation, making them well-suited for regulatory approval and ongoing monitoring. However, they may lack the ability to capture complex, non-linear relationships in the data. In contrast, complex models like Gradient Boosting often deliver higher predictive accuracy but suffer from limited interpretability, higher validation complexity, and increased governance challenges. Under Basel II, institutions must balance performance gains against the need for explainability, often favoring simpler models unless the added complexity can be clearly justified, explained, and controlled
