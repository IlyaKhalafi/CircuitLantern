import { text } from "@/i18n/format";
import { requestLanguage } from "@/i18n/server";
import { localizedMetadata } from "@/i18n/server";
import { Compass } from "lucide-react";
import { EmptyState } from "@/components/ui";

export async function generateMetadata() {
  return localizedMetadata("Page not found");
}

export default async function NotFound() {
  const { locale, messages } = await requestLanguage();
  const t = (message: string) => text(locale, messages, message);
  return (
    <div className="not-found-page">
      <EmptyState
        icon={<Compass size={28} />}
        title={t("This path is still unexplored.")}
        href="/"
        action={t("Back to the model library")}
      >
        {t(
          "There is no page at this address. Your workbench is one click away.",
        )}
      </EmptyState>
    </div>
  );
}
