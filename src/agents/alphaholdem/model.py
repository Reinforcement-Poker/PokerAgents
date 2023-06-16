import tensorflow as tf


def build_alphaholdem(n_actions: int) -> tf.keras.Model:
    actions_encoder = build_actions_encoder(n_actions)
    actions_encoding = actions_encoder.output
    actions_encoding = tf.reshape(
        actions_encoding,
        (
            tf.shape(actions_encoding)[0],
            tf.shape(actions_encoding)[1]
            * tf.shape(actions_encoding)[2]
            * tf.shape(actions_encoding)[3],
        ),
    )

    cards_encoder = build_cards_encoder()
    cards_encoding = cards_encoder.output
    cards_encoding = tf.reshape(
        cards_encoding,
        (
            tf.shape(cards_encoding)[0],
            tf.shape(cards_encoding)[1] * tf.shape(cards_encoding)[2] * tf.shape(cards_encoding)[3],
        ),
    )

    x = tf.concat([actions_encoding, cards_encoding], axis=-1)

    for i in range(1, 50, 5):
        x = tf.keras.layers.Dense((n_actions + 1) * 50 // i)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Dense(n_actions + 1)(x)
    policy = tf.keras.activations.softmax(x[:, :n_actions])  # type: ignore
    value = tf.keras.activations.sigmoid(x[:, -1]) * 200 - 100  # type: ignore

    return tf.keras.Model(
        inputs={"actions": actions_encoder.input, "cards": cards_encoder.input},
        outputs=[policy, value],
    )


def build_actions_encoder(n_actions: int) -> tf.keras.Model:
    inpt = tf.keras.Input(shape=(24, n_actions, 4))
    x = inpt

    x = tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    for i in range(5, 7):
        x = tf.keras.layers.Conv2D(filters=2**i, kernel_size=(1, 3), padding="valid")(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(filters=2**i, kernel_size=(3, 3), padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)

    return tf.keras.Model(inputs=inpt, outputs=x)


def build_cards_encoder() -> tf.keras.Model:
    inpt = tf.keras.Input(shape=(4, 13, 6))
    x = inpt

    x = tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    for i in range(5, 7):
        x = tf.keras.layers.Conv2D(filters=2**i, kernel_size=(1, 3), padding="valid")(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(filters=2**i, kernel_size=(1, 3), padding="valid")(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(filters=2**i, kernel_size=(3, 3), padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)

    return tf.keras.Model(inputs=inpt, outputs=x)
