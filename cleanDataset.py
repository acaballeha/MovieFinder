import argparse
import pandas as pd


def main(input_path, output_path):
    # Leer el archivo CSV en chunks
    chunks = pd.read_csv(input_path, encoding="utf-8", sep=",", chunksize=1000)

    # Concatenate chunks into a single DataFrame
    df_peliculas = pd.concat(chunks, ignore_index=True)

    columnas = ["title", "release_date", "popularity", "original_language", "overview", "genre_ids", "adult", "poster_path"]
    df_peliculas = df_peliculas[columnas]

    # Remove duplicate rows based on all columns
    df_peliculas.drop_duplicates(inplace=True)

    # Remove rows with missing values (NaN) in any column
    df_peliculas.dropna(inplace=True)

    # Reset the index after removing rows
    df_peliculas.reset_index(drop=True, inplace=True)

    df_peliculas.head()

    # Guardar el DataFrame limpio en un archivo CSV
    df_peliculas.to_csv(output_path, index=False, encoding="utf-8")
    print(f"✅ Archivo CSV guardado en {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and process a CSV file.")
    parser.add_argument("input_path", type=str, help="Path to the input CSV file.")

    
    args = parser.parse_args()
    inputPath= args.input_path
    outputPath = inputPath.replace(".csv", "_CLEAN.csv")
    main(inputPath, outputPath)


