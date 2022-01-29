
var kmodel = kmodel || {};
var base = base || require('./base');

kmodel.ModelFactory = class {

    match(context) {
        return kmodel.Reader.open(context.stream);
    }

    open(context, match) {
        return Promise.resolve().then(() => {
            const reader = match;
            return new kmodel.Model(reader);
        });
    }
};

kmodel.Model = class {

    constructor(model) {
        this._format = 'kmodel v' + model.version.toString();
        this._graphs = [ new kmodel.Graph(model) ];
    }

    get format() {
        return this._format;
    }

    get graphs() {
        return this._graphs;
    }
};

kmodel.Graph = class {

    constructor(model) {
        this._inputs = [];
        this._outputs = [];
        this._nodes = model.layers.map((layer) => new kmodel.Node(layer));
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get nodes() {
        return this._nodes;
    }
};

kmodel.Node = class {

    constructor(layer) {
        this._location = layer.location;
        this._type = layer.type;
    }

    get location() {
        return this._location;
    }

    get name() {
        return '';
    }

    get type() {
        return this._type;
    }

    get inputs() {
        return [];
    }

    get outputs() {
        return [];
    }

    get attributes() {
        return [];
    }
};

kmodel.Reader = class {

    static open(stream) {
        const reader = new base.BinaryReader(stream);
        if (reader.length > 4) {
            const signature = reader.uint32();
            if (signature === 3) {
                return new kmodel.Reader(reader, 3);
            }
            if (signature === 0x4B4D444C) {
                const version = reader.uint32();
                return new kmodel.Reader(reader, version);
            }
        }
        return null;
    }

    constructor(reader, version) {
        this._reader = reader;
        this._version = version;
    }

    get version() {
        return this._version;
    }

    get layers() {
        this._read();
        return this._layers;
    }

    _read() {
        if (this._reader) {
            const reader = this._reader;
            if (this._version < 3 || this._version > 5) {
                throw new kmodel.Error("Unsupported model version '" + this.version.toString() + "'.");
            }
            if (this._version === 3) {
                /* const flags = */ reader.uint32();
                /* const arch = */ reader.uint32();
                this._layers = new Array(reader.uint32());
                /* const max_start_address = */ reader.uint32();
                /* const main_mem_usage = */ reader.uint32();
                this._outputs = new Array(reader.uint32());
                for (let i = 0; i < this._outputs.length; i++) {
                    this._outputs[i] = {
                        address: reader.uint32(),
                        size: reader.uint32()
                    };
                }
                for (let i = 0; i < this._layers.length; i++) {
                    this._layers[i] = {
                        location: i,
                        type: reader.uint32(),
                        body_size: reader.uint32()
                    };
                }
                let offset = reader.position;
                for (const layer of this._layers) {
                    layer.offset = offset;
                    offset += layer.body_size;
                    // layer.body = reader.read(layer.body_size);
                    // delete layer.body_size;
                }
                const types = new Map();
                const register = (type, name, category, callback) => {
                    types.set(type, { type: { name: name, category: category || '' }, callback: callback });
                };
                register(   -1, 'DUMMY');
                register(    0, 'INVALID');
                register(    1, 'ADD');
                register(    2, 'QUANTIZED_ADD');
                register(    3, 'GLOBAL_MAX_POOL2D', 'Pool');
                register(    4, 'QUANTIZED_GLOBAL_MAX_POOL2D', 'Pool');
                register(    5, 'GLOBAL_AVERAGE_POOL2D', 'Pool', (layer, reader) => {
                    layer.flags = reader.uint32();
                    layer.main_mem_in_address = reader.uint32();
                    layer.main_mem_out_address = reader.uint32();
                    layer.kernel_size = reader.uint32();
                    layer.channels = reader.uint32();
                });
                register(    6, 'QUANTIZED_GLOBAL_AVERAGE_POOL2D', 'Pool');
                register(    7, 'MAX_POOL2D', 'Pool');
                register(    8, 'QUANTIZED_MAX_POOL2D', 'Pool', (layer, reader) => {
                    layer.flags = reader.uint32();
                    layer.main_mem_in_address = reader.uint32();
                    layer.main_mem_out_address = reader.uint32();
                    layer.in_shape = [ reader.uint32(), reader.uint32(), reader.uint32() ];
                    layer.out_shape = [ reader.uint32(), reader.uint32(), reader.uint32() ];
                    layer.kernel = [ reader.uint32(), reader.uint32() ];
                    layer.stride = [ reader.uint32(), reader.uint32() ];
                    layer.padding = [ reader.uint32(), reader.uint32() ];
                });
                register(    9, 'AVERAGE_POOL2D', 'Pool');
                register(   10, 'QUANTIZED_AVERAGE_POOL2D', 'Pool');
                register(   11, 'QUANTIZE', '', (layer, reader) => {
                    layer.flags = reader.uint32();
                    layer.main_mem_in_address = reader.uint32();
                    layer.mem_out_address = reader.uint32();
                    layer.count = reader.uint32();
                    layer.scale = reader.float32();
                    layer.bias = reader.float32();
                });
                register(   12, 'DEQUANTIZE', '', (layer, reader) => {
                    layer.flags = reader.uint32();
                    layer.main_mem_in_address = reader.uint32();
                    layer.mem_out_address = reader.uint32();
                    layer.count = reader.uint32();
                    layer.scale = reader.float32();
                    layer.bias = reader.float32();
                });
                register(   13, 'REQUANTIZE');
                register(   14, 'L2_NORMALIZATION', 'Normalization');
                register(   15, 'SOFTMAX', 'Activation', (layer, reader) => {
                    layer.flags = reader.uint32();
                    layer.main_mem_in_address = reader.uint32();
                    layer.main_mem_out_address = reader.uint32();
                    layer.channels = reader.uint32();
                });
                register(   16, 'CONCAT', 'Tensor');
                register(   17, 'QUANTIZED_CONCAT', 'Tensor');
                register(   18, 'FULLY_CONNECTED', 'Layer', (layer, reader) => {
                    layer.flags = reader.uint32();
                    layer.main_mem_in_address = reader.uint32();
                    layer.main_mem_out_address = reader.uint32();
                    layer.in_channels = reader.uint32();
                    layer.out_channels = reader.uint32();
                    layer.act = reader.uint32();
                    layer.weights = reader.read(4 * layer.in_channels * layer.out_channels);
                    layer.bias = reader.read(4 * layer.out_channels);
                });
                register(   19, 'QUANTIZED_FULLY_CONNECTED', 'Layer');
                register(   20, 'TENSORFLOW_FLATTEN', 'Shape');
                register(   21, 'QUANTIZED_TENSORFLOW_FLATTEN', 'Shape', (layer, reader) => {
                    layer.flags = reader.uint32();
                    layer.main_mem_in_address = reader.uint32();
                    layer.main_mem_out_address = reader.uint32();
                    layer.shape = [ reader.uint32(), reader.uint32(), reader.uint32() ];
                });
                register( 1000, 'CONV', 'Layer');
                register( 1001, 'DWCONV', 'Layer');
                register( 1002, 'QUANTIZED_RESHAPE', 'Shape');
                register( 1003, 'RESHAPE', 'Shape');
                register(10240, 'K210_CONV', 'Layer', (layer, reader) => {
                    layer.flags = reader.uint32();
                    layer.main_mem_out_address = reader.uint32();
                    layer.layer_offset = reader.uint32();
                    layer.weights_offset = reader.uint32();
                    layer.bn_offset = reader.uint32();
                    layer.act_offset = reader.uint32();
                });
                register(10241, 'K210_ADD_PADDING', '', (layer, reader) => {
                    layer.flags = reader.uint32();
                    layer.main_mem_in_address = reader.uint32();
                    layer.kpu_mem_out_address = reader.uint32();
                    layer.channels = reader.uint32();
                });
                register(10242, 'K210_REMOVE_PADDING', '', (layer, reader) => {
                    layer.flags = reader.uint32();
                    layer.main_mem_in_address = reader.uint32();
                    layer.kpu_mem_out_address = reader.uint32();
                    layer.channels = reader.uint32();
                });
                register(10243, 'K210_UPLOAD', '', (layer, reader) => {
                    layer.flags = reader.uint32();
                    layer.main_mem_in_address = reader.uint32();
                    layer.kpu_mem_out_address = reader.uint32();
                    layer.shape = [ reader.uint32(), reader.uint32(), reader.uint32() ];
                });
                for (const layer of this._layers) {
                    const type = types.get(layer.type);
                    if (!type) {
                        throw new kmodel.Error("Unsupported layer type '" + layer.type.toString() + "'.");
                    }
                    if (!type.callback) {
                        throw new kmodel.Error("Unsupported layer '" + type.name + "'.");
                    }
                    layer.type = type.type;
                    reader.seek(layer.offset);
                    type.callback(layer, reader);
                    delete layer.offset;
                    delete layer.body_size;
                    if (reader.position != (layer.offset + layer.body_size)) {
                        // debugger;
                    }
                    // console.log(JSON.stringify(Object.fromEntries(Object.entries(layer).filter((entry) => !(entry[1] instanceof Uint8Array))), null, 2));
                }
            }
            else if (this._version >= 4) {
                const header_size = reader.uint32();
                /* const flags = */ reader.uint32();
                /* const alignment = */ reader.uint32();
                this.modules = new Array(reader.uint32());
                /* const entry_module = */ reader.uint32();
                /* const entry_function = */ reader.uint32();
                if (header_size > reader.position) {
                    reader.skip(header_size - reader.position);
                }
                for (let i = 0; i < this.modules.length; i++) {
                    /*
                    char[16] type;
                    uint32_t version;
                    uint32_t header_size;
                    uint32_t size;
                    uint32_t mempools;
                    uint32_t shared_mempools;
                    uint32_t sections;
                    uint32_t functions;
                    uint32_t reserved0;
                    */
                }
                throw new kmodel.Error("Unsupported model version '" + this.version.toString() + "'.");
            }
            delete this._reader;
        }
    }
};

kmodel.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading kmodel.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = kmodel.ModelFactory;
}