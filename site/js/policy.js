export function createPolicy(bytes, observations) {
  return {
    act: indices => indices.map((index, agent) => bytes[agent * observations + index]),
  };
}
