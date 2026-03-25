import "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js";
import { workerEvents } from "../events/constants.js";

console.log("Model training worker initialized");
let _globalCtx = {};
let _model = null;
const WEIGHTS = {
  category: 0.4,
  color: 0.3,
  price: 0.2,
  age: 0.1,
};

const normalize = (value, min, max) => (value - min) / (max - min) || 1;
function makeContext(products, users) {
  const ages = users.map((u) => u.age);
  const prices = products.map((p) => p.price);
  const minAge = Math.min(...ages);
  const maxAge = Math.max(...ages);
  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);
  const colors = [...new Set(products.map((p) => p.color))];
  const categories = [...new Set(products.map((p) => p.category))];
  const colorIndex = Object.fromEntries(
    colors.map((color, index) => {
      return [color, index];
    }),
  );
  const categoriesIndex = Object.fromEntries(
    categories.map((category, index) => {
      return [category, index];
    }),
  );
  //Computar a média de idade dos compradores por produto
  //ajuda personalizada
  const midAge = (minAge + maxAge) / 2;
  const ageSums = {};
  const ageCounts = {};

  users.forEach((user) => {
    user.purchases.forEach((p) => {
      ageSums[p.name] = (ageSums[p.name] || 0) + user.age;
      ageCounts[p.name] = (ageCounts[p.name] || 0) + 1;
    });
  });

  const productsAvgAgeNorm = Object.fromEntries(
    products.map((product) => {
      const avg = ageCounts[product.name]
        ? ageSums[product.name] / ageCounts[product.name]
        : midAge;
      return [product.name, normalize(avg, minAge, maxAge)];
    }),
  );
  return {
    products,
    users,
    colors,
    colorIndex,
    categoriesIndex,
    minAge,
    maxAge,
    minPrice,
    maxPrice,
    numCategories: categories.length,
    numColors: colors.length,
    productsAvgAgeNorm,
    //price + age + category + color
    dimentions: 2 + categories.length + colors.length,
  };
}
function oneHotWeighted(index, length, weight) {
  return tf.oneHot(index, length).cast("float32").mul(weight);
}
function encodeProduct(product, context) {
  const price = tf.tensor1d([
    normalize(product.price, context.minPrice, context.maxPrice) *
      WEIGHTS.price,
  ]);
  const age = tf.tensor1d([
    (context.productsAvgAgeNorm[product.name] ?? 0.5) * WEIGHTS.age,
  ]);

  const category = oneHotWeighted(
    context.categoriesIndex[product.category],
    context.numCategories,
    WEIGHTS.category,
  );

  const color = oneHotWeighted(
    context.colorIndex[product.color],
    context.numColors,
    WEIGHTS.color,
  );

  return tf.concat1d([price, age, category, color]);
}

function encodeUser(user, context) {
  if (user.purchases.length) {
    return tf
      .stack(user.purchases.map((product) => encodeProduct(product, context)))
      .mean(0)
      .reshape([1, context.dimentions]);
  }

  return tf
    .concat1d([
      tf.zeros([1]), // preço ignorado
      tf.tensor1d([
        normalize(user.age, context.minAge, context.maxAge) * WEIGHTS.age,
      ]),
      tf.zeros([context.numCategories]), // categoria ignorada
      tf.zeros([context.numColors]), // cor ignorada
    ])
    .reshape([1, context.dimentions]);
}

function createTrainingData(context) {
  const inputs = [];
  const labels = [];
  context.users
    .filter((user) => user.purchases.length > 0)
    .forEach((user) => {
      const userVector = encodeUser(user, context).dataSync();
      context.products.forEach((product) => {
        const productVector = encodeProduct(product, context).dataSync();
        const label = user.purchases.some((purchase) =>
          purchase.name === product.name ? 1 : 0,
        );
        //combinar usuario e produto
        inputs.push([...userVector, ...productVector]);
        labels.push(label);
      });
    });
  return {
    xs: tf.tensor2d(inputs),
    ys: tf.tensor2d(labels, [labels.length, 1]),
    inputDimensions: context.dimentions * 2,
    //tamanho= userVector + productVector
  };
}

async function configureNeuralNetAndTrain(trainData) {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [trainData.inputDimensions],
      units: 128,
      activation: "relu",
    }),
  );
  //Camada oculta 1
  // - 64 neurônios (menos que a primeira camada: começa a comprimir a informação)
  // - activation: "relu" (função de ativação: introduz não-linearidade)

  model.add(
    tf.layers.dense({
      units: 64,
      activation: "relu",
    }),
  );
  //Camada oculta 2
  // - 32 neurônios (menos que a primeira camada: começa a comprimir a informação)
  // - activation: "relu" (função de ativação: introduz não-linearidade)
  model.add(
    tf.layers.dense({
      units: 32,
      activation: "relu",
    }),
  );
  //Camada de saída
  // - 1 neurônio (saída binária: 0 ou 1)
  // - activation: "sigmoid" (função de ativação: introduz não-linearidade)
  // - Exemplo: se a saída for 0.8, significa que há 80% de chance de o usuário comprar o produto
  model.add(
    tf.layers.dense({
      units: 1,
      activation: "sigmoid",
    }),
  );
  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });
  await model.fit(trainData.xs, trainData.ys, {
    epochs: 100,
    batchSize: 32,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        postMessage({
          type: workerEvents.trainingLog,
          epoch: epoch,
          loss: logs.loss,
          accuracy: logs.acc,
        });
      },
    },
  });
  return model;
}

async function trainModel({ users }) {
  console.log("Training model with users:", users);

  postMessage({
    type: workerEvents.progressUpdate,
    progress: { progress: 50 },
  });
  const products = await (await fetch("/data/products.json")).json();

  const context = makeContext(products, users);
  context.productVectors = products.map((product) => {
    return {
      name: product.name,
      metadata: { ...product },
      vector: encodeProduct(product, context).dataSync(),
    };
  });

  _globalCtx.context = context;

  const trainData = createTrainingData(context);
  _model = await configureNeuralNetAndTrain(trainData);

  postMessage({
    type: workerEvents.progressUpdate,
    progress: { progress: 100 },
  });
  postMessage({ type: workerEvents.trainingComplete });
}
function recommend(user) {
  if (!_model) return;
  const context = _globalCtx.context;
  //Converta o usuário fornecido no vetor de features codificadas
  //preço ignorado
  //idade normalizada
  //categoria ignorada

  const userVector = encodeUser(user, context).dataSync();
  //Em aplicações reais:
  // Armazene todos os vetores de produtos em um banco de dados vetorial (ex: Pinecone, Weaviate, Milvus)
  // para permitir busca rápida de similaridade.
  //Consulta: Encontre os 200 produtos mais similares ao vetor do usuário.
  //Execute _model.predict() apenas nos 200 vetores de produtos mais similares.
  //cria um array de vetores de entrada combinando o vetor do usuário com o vetor de cada produto
  const inputs = context.productVectors.map(({ vector }) => {
    return [...userVector, ...vector];
  });
  const inputTensor = tf.tensor2d(inputs);
  const scores = _model.predict(inputTensor).dataSync();
  const results = context.productVectors.map((product, index) => {
    return {
      ...product.metadata,
      name: product.name,
      score: scores[index], //previsão de compra
    };
  });
  const sortedItems = results.sort((a, b) => b.score - a.score);
  postMessage({
    type: workerEvents.recommend,
    user,
    recommendations: sortedItems,
  });
}

const handlers = {
  [workerEvents.trainModel]: trainModel,
  [workerEvents.recommend]: (d) => recommend(d.user, _globalCtx),
};

self.onmessage = (e) => {
  const { action, ...data } = e.data;
  if (handlers[action]) handlers[action](data);
};
