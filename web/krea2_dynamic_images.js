import { app } from "../../scripts/app.js";

// Auto-growing reference slots for "Text Encode (Krea2)": each reference is an
// (imageN, maskN) pair. Keep exactly one empty trailing pair, appending a fresh
// pair whenever the last one gets connected.
const NODE_NAMES = new Set(["TextEncodeKrea2"]);
const IMAGE_RE = /^image(\d+)$/;

function pairNumbers(node) {
    const nums = [];
    for (const inp of node.inputs || []) {
        const m = IMAGE_RE.exec(inp.name);
        if (m) nums.push(parseInt(m[1], 10));
    }
    nums.sort((a, b) => a - b);
    return nums;
}

function inputIndex(node, name) {
    return (node.inputs || []).findIndex((i) => i.name === name);
}

function linked(node, name) {
    const idx = inputIndex(node, name);
    return idx >= 0 && node.inputs[idx].link != null;
}

function addPair(node, n) {
    node.addInput(`image${n}`, "IMAGE");
    node.addInput(`mask${n}`, "MASK");
}

function removePair(node, n) {
    // Remove mask first (it sits after the image, so the image index stays valid).
    let idx = inputIndex(node, `mask${n}`);
    if (idx >= 0) node.removeInput(idx);
    idx = inputIndex(node, `image${n}`);
    if (idx >= 0) node.removeInput(idx);
}

function pairEmpty(node, n) {
    return !linked(node, `image${n}`) && !linked(node, `mask${n}`);
}

// Self-heal: guarantee every imageN has a companion maskN (e.g. nodes saved before
// masks existed, or a stale Python schema that only defined image1).
function ensureMasks(node) {
    for (const n of pairNumbers(node)) {
        if (inputIndex(node, `mask${n}`) < 0) {
            node.addInput(`mask${n}`, "MASK");
        }
    }
}

function syncPairs(node) {
    if (pairNumbers(node).length === 0) addPair(node, 1);
    ensureMasks(node);

    // Collapse trailing fully-empty pairs down to a single spare.
    for (;;) {
        const nums = pairNumbers(node);
        if (nums.length <= 1) break;
        const last = nums[nums.length - 1];
        const prev = nums[nums.length - 2];
        if (pairEmpty(node, last) && pairEmpty(node, prev)) {
            removePair(node, last);
        } else {
            break;
        }
    }

    // If the last pair is in use, append a fresh spare pair.
    const nums = pairNumbers(node);
    const last = nums[nums.length - 1];
    if (!pairEmpty(node, last)) {
        addPair(node, last + 1);
    }

    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "krea2.textencode.dynamicimages",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_NAMES.has(nodeData.name)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            syncPairs(this);
            return r;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (slotType) {
            const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
            // LiteGraph.INPUT === 1; only react to input-side connection changes.
            if (slotType === 1) {
                syncPairs(this);
            }
            return r;
        };

        // Restore the spare pair after a saved graph is loaded.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            syncPairs(this);
            return r;
        };
    },
});
