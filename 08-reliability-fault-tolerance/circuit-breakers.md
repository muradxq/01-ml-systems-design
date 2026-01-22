# Circuit Breakers

## Overview

Circuit breakers prevent cascading failures by stopping requests to failing services and allowing recovery.

---

## 🎯 States

### 1. Closed (Normal)
- Requests pass through
- Monitor failures
- Count errors

### 2. Open (Failing)
- Requests fail fast
- No calls to service
- Allow recovery time

### 3. Half-Open (Testing)
- Test service recovery
- Limited requests
- Transition based on results

---

## ✅ Best Practices

1. **Set thresholds** - error rates, timeouts
2. **Monitor state** - track transitions
3. **Test recovery** - ensure it works
4. **Log transitions** - for debugging
5. **Tune parameters** - based on behavior

---

## 🔗 Related Topics

- [High Availability](./high-availability.md)
- [Graceful Degradation](./graceful-degradation.md)
