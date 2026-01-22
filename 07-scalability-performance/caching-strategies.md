# Caching Strategies

## Overview

Caching reduces latency and computation by storing frequently accessed data and predictions.

---

## 🎯 Caching Patterns

### 1. Prediction Caching
- Cache model predictions
- Key: input features
- TTL: based on freshness needs

### 2. Feature Caching
- Cache computed features
- Reduce feature computation
- Faster inference

### 3. Model Caching
- Cache loaded models
- Reduce model loading time
- Memory optimization

---

## ✅ Best Practices

1. **Cache predictions** - for repeated inputs
2. **Cache features** - reduce computation
3. **Set TTLs** - balance freshness and performance
4. **Monitor cache** - hit rates, performance
5. **Invalidate appropriately** - when data changes

---

## 🔗 Related Topics

- [Horizontal Scaling](./horizontal-scaling.md)
- [Optimization Techniques](./optimization-techniques.md)
