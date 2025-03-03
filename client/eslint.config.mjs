import { dirname } from "path";
import { fileURLToPath } from "url";
import { FlatCompat } from "@eslint/eslintrc";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const compat = new FlatCompat({
  baseDirectory: __dirname,
});

const eslintConfig = [
  // This empty config will effectively disable all rules
  { 
    linterOptions: {
      noInlineConfig: false,
    },
    rules: {
      // Set all rules to "off"
      // This overrides any rules from the extended configs
      "@next/next/no-html-link-for-pages": "off",
      "react/no-unescaped-entities": "off",
      // Add any other specific rules you want to disable
    }
  }
  
  // Comment out the extended configs to completely disable them
  // ...compat.extends("next/core-web-vitals", "next/typescript"),
];

export default eslintConfig;
