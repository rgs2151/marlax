export function createPolicy(bytes, observations, {epsilon = 0, numActions = 5, random = Math.random} = {}) {
  return {
    act: indices => indices.map((index, agent) =>
      epsilon > 0 && random() < epsilon
        ? Math.floor(random() * numActions)
        : bytes[agent * observations + index]),
  };
}
