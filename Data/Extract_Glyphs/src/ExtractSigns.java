import jsesh.hieroglyphs.data.HieroglyphDatabaseRepository;
import jsesh.hieroglyphs.data.HieroglyphDatabaseInterface;

import java.io.*;
import java.util.*;

public class ExtractSigns {

    static boolean isValidGardnerCode(String code) {
        if (code == null || code.isEmpty()) return false;
        char first = code.charAt(0);
        return first >= 'A' && first <= 'Z';
    }

    static String classifySign(List<String> values) {
        if (values == null || values.isEmpty()) {
            return "Determinative";
        }
        int minLength = Integer.MAX_VALUE;
        for (String v : values) {
            if (v.length() < minLength) minLength = v.length();
        }
        if (minLength <= 2) {
            return "Phonetic";
        }
        return "Ideogram";
    }

    public static void main(String[] args) throws Exception {

        HieroglyphDatabaseInterface db = HieroglyphDatabaseRepository.getHieroglyphDatabase();

        PrintWriter writer = new PrintWriter(new OutputStreamWriter(
                new FileOutputStream("signs.csv"), "UTF-8"));

        writer.println("code,transliterations,type");

        int valid = 0, skipped = 0;

        for (String code : db.getCodesSet()) {

            if (!isValidGardnerCode(code)) {
                skipped++;
                continue;
            }

            List<String> values = db.getValuesFor(code);
            String joined = (values != null && !values.isEmpty()) ? String.join("|", values) : "";
            String type = classifySign(values);

            writer.println(code + ",\"" + joined + "\",\"" + type + "\"");
            valid++;
        }

        writer.close();
        System.out.println("Done! File saved: signs.csv");
        System.out.println("Valid Gardner codes: " + valid);
        System.out.println("Skipped invalid codes: " + skipped);
    }
}