1. If you only had 200 labeled replies, how would you improve the model without collecting thousands more?

I’d use data augmentation and semi-supervised learning to expand the dataset. Pretrained embeddings or transfer learning can also help the model generalize better with fewer labels.

2. How would you ensure your reply classifier doesn’t produce biased or unsafe outputs in production?

I will integrate an AI-based content moderation layer that flags unsafe or biased replies before they reach users.

3. Suppose you want to generate personalized cold email openers using an LLM. What prompt design strategies would you use to keep outputs relevant and non-generic?

I will include context like the recipient’s role, company, or recent news in the prompt. Adding clear instructions to keep the tone professional and concise makes the output more personalized.
