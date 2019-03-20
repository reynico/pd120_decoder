'''
Punto de entrada al programa.

Leer argumentos de entrada y llamar a recepcion.main(args) o transmision(args)

Argumentos leidos en RX y TX:

- `args.rf_freq`: float.
- `args.rf_rate`: float.
- `args.audio_rate`: float.

Argumentos leidos en TX:

- `args.image`: str.
- `args.sstv_wav`: str o None si no se guarda el WAV.
'''

import argparse
import recepcion
import transmision

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='comandos', dest='comando')
subparsers.required = True

# Argumentos RX y TX

parser.add_argument(
        '-f', '--rf-freq', type=float,
        help=('Frecuencia de transmision o recepción de RF en Hz, por defecto: '
              '137.2e6 para trabajar en los 137.2MHz'),
        default=137.2e6,
)
parser.add_argument(
        '--rf-rate', type=float,
        help=('Tasa de muestreo a usar al trabajar en RF, escribir en Hz, por '
              'defecto: 300e3 para usar el SDR con en los 300ksps. Ver en GQRX '
              'las tasas posibles, creo que son 100ksps, 300ksps, 960ksps, y '
              'mas de eso no haría falta. Debe ser múltiplo de --audio-rate'),
        default=300e3,
)
parser.add_argument(
        '--audio-rate', type=float,
        help=('Tasa de muestreo a usar al trabajar en audio, escribir en Hz, '
              'por defecto: 30e3 para trabajar con audios a 30kHz. Usar tasas '
              'que sean fracciones enteras de --rf-rate'),
        default=300e3,
)

# Argumentos RX

parser_rx = subparsers.add_parser('rx', help='recibir SSTV')

# Argumentos TX

parser_tx = subparsers.add_parser('tx', help='transmitir SSTV')
parser_tx.add_argument(
        'image', type=str,
        help=('Nombre de imagen a transmitir, debe ser un PNG'),
)
parser_tx.add_argument(
        '--sstv-wav', type=str,
        help=('Nombre archivo WAV en donde guardar el SSTV generado, opcional'),
        default=None,
)

#

args = parser.parse_args()

if args.comando == 'rx':
    recepcion.main(args)
elif args.comando == 'tx':
    transmision.main(args)
else:
    RuntimeError('Comando desconocido')

