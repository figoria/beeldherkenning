export default class kNear {
    constructor(k) {
        this.k = k;
        this.training = [];
    }

    learn(vector, label) {
        this.training.push({ vector, label });
    }

    _distance(vec1, vec2) {
        return Math.sqrt(vec1.reduce((sum, value, index) => {
            return sum + Math.pow(value - vec2[index], 2);
        }, 0));
    }

    classify(observation) {
        const distances = this.training.map((example) => ({
            label: example.label,
            distance: this._distance(example.vector, observation),
        }));

        distances.sort((a, b) => a.distance - b.distance);

        const nearest = distances.slice(0, this.k);
        const frequency = nearest.reduce((freq, example) => {
            freq[example.label] = (freq[example.label] || 0) + 1;
            return freq;
        }, {});

        const avgDistance = nearest.reduce((sum, example) => sum + example.distance, 0) / nearest.length;

        return {
            label: Object.keys(frequency).reduce((a, b) => frequency[a] > frequency[b] ? a : b),
            avgDistance: avgDistance,
        };
    }
}
