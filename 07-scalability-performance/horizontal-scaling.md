# Horizontal Scaling

## Overview

Horizontal scaling adds more instances to handle increased load. It's essential for ML systems that need to serve many requests.

---

## 🎯 Scaling Strategies

### 1. Auto-scaling
- Scale based on metrics
- CPU, memory, request rate
- Automatic adjustment

### 2. Load Balancing
- Distribute requests
- Health checks
- Failover

### 3. Stateless Design
- No shared state
- Any instance can handle request
- Easy to scale

---

## ✅ Best Practices

1. **Design for scale** - stateless, distributed
2. **Use auto-scaling** - automatic adjustment
3. **Load balance** - distribute traffic
4. **Monitor scaling** - track metrics
5. **Test scaling** - load testing

---

## 🔗 Related Topics

- [Caching Strategies](./caching-strategies.md)
- [Optimization Techniques](./optimization-techniques.md)
