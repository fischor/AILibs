package jaicore.ml.tsc.latexeval;

import java.util.HashMap;
import java.util.Map;

import jaicore.basic.SQLAdapter;
import jaicore.basic.kvstore.KVStore;
import jaicore.basic.kvstore.KVStoreCollection;
import jaicore.basic.kvstore.KVStoreCollection.EGroupMethod;
import jaicore.basic.kvstore.KVStoreUtil;

public class ResultCollector {

	public static void main(final String[] args) throws Exception {

		Map<String, String> commonFields = new HashMap<>();

		{
			// TODO: host, user, password, database
			try (SQLAdapter adapter = new SQLAdapter("", "", "", "tsc")) {
				KVStoreCollection csvChunks = KVStoreUtil.readFromMySQLTable(adapter, "",
						commonFields);
				
				KVStoreCollection newStore = new KVStoreCollection();
				
				for (KVStore k : csvChunks) {
					if (k.getAsString("accuracy") == null || k.getAsString("ref_accuracy") == null)
						continue;

					KVStore ownK = new KVStore();
					KVStore refK = new KVStore();
					ownK.put("impl", "own");
					ownK.put("acc", Double.parseDouble(k.getAsString("accuracy")));
					ownK.put("train_time", Double.parseDouble(k.getAsString("train_time")));
					
					refK.put("impl", "ref");
					refK.put("acc", Double.parseDouble(k.getAsString("ref_accuracy")));
					refK.put("train_time", Double.parseDouble(k.getAsString("ref_train_time")));
					
					ownK.put("dataset", k.getAsString("dataset"));
					refK.put("dataset", k.getAsString("dataset"));

					newStore.add(ownK);
					newStore.add(refK);
				}

				HashMap<String, EGroupMethod> handler = new HashMap<String, EGroupMethod>();
				handler.put("acc", EGroupMethod.AVG);
				handler.put("train_time", EGroupMethod.AVG);
				newStore = newStore.group(new String[] { "impl", "dataset" }, handler);

				System.out.println("===== ACCURACIES =====");

				for (KVStoreCollection c : new KVStoreCollection[] { newStore }) {
					String latexTable = KVStoreUtil.kvStoreCollectionToLaTeXTable(c, "dataset", "impl",
							"acc",
							"-$\\phantom{\\bullet}$");
					System.out.println(latexTable);
				}

				System.out.println("===== TRAIN TIME =====");

				for (KVStoreCollection c : new KVStoreCollection[] { newStore }) {
					String latexTable = KVStoreUtil.kvStoreCollectionToLaTeXTable(c, "dataset", "impl", "train_time",
							"-$\\phantom{\\bullet}$");
					System.out.println(latexTable);
				}
			}
		}

	}

}
