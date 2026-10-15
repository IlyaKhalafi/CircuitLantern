import { Compass } from "lucide-react";
import { EmptyState } from "@/components/ui";

export default function NotFound() {
  return (
    <EmptyState
      icon={<Compass size={28} />}
      title="This path is still unexplored."
      href="/"
      action="Back to the model library"
    >
      There is no page at this address. Your workbench is one click away.
    </EmptyState>
  );
}
