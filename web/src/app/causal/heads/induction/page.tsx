import { InductionWorkbench } from "@/components/induction-workbench";
import { localizedMetadata } from "@/i18n/server";
import "./induction.css";

export async function generateMetadata() {
  const { title } = await localizedMetadata("Induction probe");
  return { title: { absolute: title + " · Circuit Lantern" } };
}

export default function InductionPage() {
  return <InductionWorkbench />;
}
