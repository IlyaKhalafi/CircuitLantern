import { CrosscoderWorkbench } from "@/components/crosscoder-workbench";
import { localizedMetadata } from "@/i18n/server";
import "./compare.css";

export async function generateMetadata() {
  return localizedMetadata("Shared feature comparisons");
}
export default function Page() {
  return <CrosscoderWorkbench />;
}
